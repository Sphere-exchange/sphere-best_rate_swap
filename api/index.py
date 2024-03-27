from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
from scipy.optimize import minimize
import time
from decimal import Decimal, ROUND_DOWN
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route('/')
def hello_world():
    return 'Hello, World! BestRateRouter is Healthy'

all_edges = []
edges_orderbook = {}
MIN_BALANCE_THRESHOLD = 0.0000001

class TradingNode:
    """
    Represents a node in the trading graph, corresponding to a specific token.
    """
    def __init__(self, token, amount=0):
        self.token = token
        self.amount = amount
        self.edges = {}  # Dictionary mapping next tokens to their reserves
        self.distributed = {}  # Dictionary mapping next tokens to their distributed amounts

    def add_edge(self, next_token, edge_data):
        """
        Adds an edge to the node, connecting it to another token.
        """
        if next_token not in self.edges:
            self.edges[next_token] = []
        self.edges[next_token].append(edge_data)

    def add_distribute(self, next_token, distribute_amount, pool_id):
        """
        Adds distributed amounts to the node for a specific next token.
        """
        if next_token not in self.distributed:
            self.distributed[next_token] = []

        temp_distributed = {}
        for dist in self.distributed[next_token]:
            temp_distributed[dist[0]] = dist[1]

        for i, pid in enumerate(pool_id):
            if pid in temp_distributed:
                temp_distributed[pid] += distribute_amount[i]
            else:
                temp_distributed[pid] = distribute_amount[i]

        self.distributed[next_token] = [[pid, amount] for pid, amount in temp_distributed.items()]

    def update_amount(self, new_amount):
        """
        Updates the amount of the token in the node.
        """
        self.amount += new_amount

    def distribute_amount(self, next_token, distribute_amounts, pool_id):
        """
        Distributes amounts to the next token and updates the node's distributed amounts.
        """
        total_distributed = sum(distribute_amounts)
        self.amount -= total_distributed
        self.add_distribute(next_token, distribute_amounts, pool_id)

    def update_reserves(self, next_token, pool_id, sold_amount):
        """
        Updates the reserves of the node based on the sold amount.
        """
        if sold_amount == 0:
            return 0
        for edge in self.edges[next_token]:
            if edge.get('pool_id') == pool_id:
                X, Y = edge['reserves'][0], edge['reserves'][1]  # Current reserves
                K = X * Y  # Constant product
                X_new = X + sold_amount  # Update reserve of sold token
                Y_new = K / X_new  # Calculate new reserve of bought token using constant product formula
                bought_amount = Y - Y_new  # Calculate bought amount

                edge['reserves'][0], edge['reserves'][1] = X_new, Y_new
                return bought_amount

        return None

class NodeManager:
    """
    Manages the nodes in the trading graph.
    """
    def __init__(self):
        self.nodes = {}

    def add_or_update_node(self, token, amount=0, edge_data=None, next_token=None):
        """
        Adds or updates a node in the node manager.
        """
        if token not in self.nodes:
            self.nodes[token] = TradingNode(token)

        if edge_data is not None and next_token is not None:
            self.nodes[token].add_edge(next_token, edge_data)
        
        self.nodes[token].update_amount(amount)

    def get_node(self, token):
        """
        Retrieves a node from the node manager based on the token.
        """
        return self.nodes.get(token, None)

def optimize_trade(liquidity_sources, sell_amount, min_trade_amount=0.001):
    """
    Optimizes the trade amounts based on the liquidity sources and sell amount.
    """
    sell_amount = Decimal(sell_amount)
    min_trade_amount = Decimal(str(min_trade_amount))
    index_orderbook = None
    if any(source['type'] == 'orderbook' for source in liquidity_sources):
        total_available_in_orderbook = sum(
            Decimal(amount) for source in liquidity_sources if source['type'] == 'orderbook' for _, amount in source['data']
        )
    else:
        total_available_in_orderbook = Decimal('0')
    for index, item in enumerate(liquidity_sources):
        if item['type'] == 'orderbook':
            index_orderbook = index
            break
        index_orderbook = None

    def objective(trade_amounts):
        """
        Objective function to maximize the amount of token B received.
        """
        total_b_received = Decimal('0')
        
        for source, trade_amount in zip(liquidity_sources, trade_amounts):
            trade_amount = Decimal(trade_amount)
            data = source['data']

            if source['type'] == 'amm':
                if isinstance(data, list):
                    x, y, _ = data
                    x, y = Decimal(x), Decimal(y)
                    delta_b = y - (x * y) / (x + trade_amount)
                    total_b_received += delta_b
                elif callable(data):
                    total_b_received += data(trade_amount)
                else:
                    print(f"{source} Data type is unknown.")
                    exit()
            elif source['type'] == 'orderbook':
                total_b_received += Decimal(calculate_orderbook(data, trade_amount))

        return -float(total_b_received)

    def constraint(trade_amounts):
        """
        Constraint function to ensure the sum of trade amounts equals the sell amount.
        """
        return float(sum(Decimal(amount) for amount in trade_amounts) - sell_amount)

    def orderbook_constraint(trade_amounts):
        """
        Constraint function for the order book trade amounts.
        """
        return float(total_available_in_orderbook) - trade_amounts[index_orderbook]
    
    initial_guess = [float(sell_amount / len(liquidity_sources)) for _ in range(len(liquidity_sources))]
    bounds = [(0, float(sell_amount)) for _ in range(len(liquidity_sources))]
    cons = [{'type': 'eq', 'fun': constraint}]

    if index_orderbook is not None:
        orderbook_cons = {'type': 'ineq', 'fun': lambda trade_amounts: float(total_available_in_orderbook) - float(trade_amounts[index_orderbook])}
        cons.append(orderbook_cons)

    # Perform optimization using SciPy's minimize function
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000})
    print("\n")
    print("total_available_in_orderbook",total_available_in_orderbook)
    print("sell_amount",sell_amount)
    print("initial_guess",initial_guess)
    print("bounds",bounds)
    print("cons",cons)
    print(result)
    optimized_trade_amounts = result.x
    optimized_trade_amounts_decimal = [Decimal(amount) for amount in optimized_trade_amounts]

    # Adjust trade amounts below the minimum threshold and redistribute the surplus
    adjusted_trade_amounts = [amount if amount >= min_trade_amount else Decimal('0') for amount in optimized_trade_amounts_decimal]
    surplus = sum(optimized_trade_amounts_decimal) - sum(adjusted_trade_amounts)
    trades_above_threshold = sum(1 for amount in adjusted_trade_amounts if amount > 0)

    if trades_above_threshold > 0:
        additional_amount = surplus / trades_above_threshold
        adjusted_trade_amounts = [amount + additional_amount if amount > 0 else Decimal('0') for amount in adjusted_trade_amounts]

    format_adjusted_trade_amounts = [float(amount.quantize(Decimal('.00000001'), rounding=ROUND_DOWN)) for amount in adjusted_trade_amounts]
    print("format_adjusted_trade_amounts", format_adjusted_trade_amounts)
    if not result.success:
        print(f"Warning: Optimization did not fully converge: {result.message}. Using the best found solution.")

    return format_adjusted_trade_amounts

def find_average_reserves(token, edges, average_reserves_cache):
    """
    Finds the average reserves for a given token pair.
    """
    cache_key = f"{token}"
    
    if cache_key in average_reserves_cache:
        return average_reserves_cache[cache_key]
    
    total_reserves_token = 0
    total_reserves_usdt = 0
    count = 0
    
    for edge in edges:
        if (edge['token'] == token and edge['next_token'] == '0xf9529e6c0951efa422d96ca39f8ee582054fd55d') or (edge['token'] == '0xf9529e6c0951efa422d96ca39f8ee582054fd55d' and edge['next_token'] == token):
            for reserves in edge['reserves']:
                if edge['token'] == token:
                    total_reserves_token += reserves[0]
                    total_reserves_usdt += reserves[1]
                else:
                    total_reserves_token += reserves[1]
                    total_reserves_usdt += reserves[0]
                count += 1
    
    if count == 0:
        raise ValueError(f"No reserves found for {token}-USDT pairs.")
    
    average_reserves = [total_reserves_token / count, total_reserves_usdt / count]
    average_reserves_cache[cache_key] = average_reserves
    
    return average_reserves

def get_asset_to_usdt_converter(asset_reserves, asset_usdt_reserves):
    """
    Returns a function that converts an asset amount to USDT equivalent.
    """
    R_asset, R_intermediate, k = asset_reserves
    R_intermediate_usdt, R_USDT = asset_usdt_reserves
    
    def converter(asset_amount):
        O_intermediate = Decimal(R_intermediate) - (Decimal(R_intermediate) * Decimal(R_asset)) / (Decimal(R_asset) + asset_amount)
        O_USDT = Decimal(R_USDT) - (Decimal(R_USDT) * Decimal(R_intermediate_usdt)) / (Decimal(R_intermediate_usdt) + O_intermediate)
        
        return O_USDT
    
    return converter

def find_paths(start_token, destination_token, visited=None, path=[]):
    """
    Finds all possible paths from the start token to the destination token.
    """
    if visited is None:
        visited = set()
    
    visited.add(start_token)

    if start_token == destination_token:
        return [path]

    paths = []
    for edge in all_edges:
        if edge['from'] == start_token and edge['to'] not in visited:
            new_path = path + [edge]
            new_paths = find_paths(edge['to'], destination_token, visited.copy(), new_path)
            paths.extend(new_paths)

    return paths

def format_paths_as_graph(paths):
    """
    Formats the paths into a graph representation.
    """
    token_pair_to_pools = {}

    for path in paths:
        for edge in path:
            token_pair = (edge['from'], edge['to'])

            if token_pair not in token_pair_to_pools:
                token_pair_to_pools[token_pair] = []

            pool_info = (edge['pool_id'], tuple(edge['reserves']), edge['type'])

            if pool_info not in token_pair_to_pools[token_pair]:
                token_pair_to_pools[token_pair].append(pool_info)

    graph_representation = []
    for (from_token, to_token), pools in token_pair_to_pools.items():
        pool_ids = [pool[0] for pool in pools]
        reserves = [pool[1] for pool in pools]
        types = [pool[2] for pool in pools]
        graph_representation.append({
            "token": from_token,
            "next_token": to_token,
            "pool_ids": pool_ids,
            "reserves": reserves,
            "types": types,
        })

    return graph_representation

def transform_to_new_format(token_distribute_node):
    """
    Transforms the token distribution node to a new format.
    """
    amm_trades = []
    orderbook_amounts = []

    for addressIn, targets in token_distribute_node.items():
        for addressOut, trades in targets.items():
            for trade in trades:
                pool, amount = trade
                rounded_amount = round(amount, 6)
                if pool != 'None':
                    amm_trades.append([addressIn, addressOut, rounded_amount, pool])
                else:
                    orderbook_amounts.append(rounded_amount)

    final_structure = {'amm': amm_trades, 'orderbook': orderbook_amounts}
    return final_structure

def fetch_edges():
    """
    Fetches the edges (token pairs) from the API endpoint.
    """
    url = "https://sphere-router.onrender.com/admin/liquidityPool/all"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    for entry in data['entries']:
        edge = {
            'type': 'amm',
            'from': entry['token0_address'].lower(),
            'to': entry['token1_address'].lower(),
            'pool_id': entry['pool_address'],
            'reserves': [float(entry['token0_reserve']), float(entry['token1_reserve'])]
        }
        all_edges.append(edge)
    
    return all_edges

def add_orderbook_edge(node_manager, edge_orderbook):
    """
    Adds an order book edge to the node manager.
    """
    token = edge_orderbook["from"]
    next_token = edge_orderbook["to"]
    
    edge_data = {
        'type': edge_orderbook['type'],
        'order': edge_orderbook['order'],
        'pool_id': edge_orderbook.get('pool_id')
    }
    node_manager.add_or_update_node(token, edge_data=edge_data, next_token=next_token)

def prepare_data_for_optimization(node, average_reserves_cache):
    """
    Prepares the liquidity sources data for optimization.
    """
    liquidity_sources = []
    for next_token, edges in node.edges.items():
        for edge in edges:
            print(edge)
            if edge['type'] == 'amm':
                if next_token != '0xf9529e6c0951efa422d96ca39f8ee582054fd55d':
                    print(next_token,"average_reserves_cache",average_reserves_cache[next_token])
                    if node.token == "0xf9529e6c0951efa422d96ca39f8ee582054fd55d":
                        tmp_x,tmp_y,tmp_k =edge['reserves']
                        liquidity_sources.append({
                            'type': edge['type'],
                            'data': [tmp_y,tmp_x,tmp_k],
                            'pool_id': edge['pool_id']
                        }) 
                    else:
                            try:
                                data = get_asset_to_usdt_converter(edge['reserves'], average_reserves_cache[next_token])
                                liquidity_sources.append({
                                    'type': edge['type'],
                                    'data': data,
                                    'pool_id': edge['pool_id']
                                })
                            except Exception as e:
                                # THIS CASE SHALL WORK WHEN WE HAVE ONLY 1 NEXT TOKEN , SO WE DIDN'T HAVE TO COMPARE ANY THING.
                                # THIS CASE IS WHEN THE NEXT TOKEN IS NOT HAVE ANY RELATION WITH THE USDT TOKEN
                                # FOR THE 1 - 1 RELATION BETWEEN THE TOKEN , WHICH IN THE REAL SCENARIO IS NOT POSSIBLE.
                                liquidity_sources.append({
                                    'type': edge['type'],
                                    'data': edge['reserves'],
                                    'pool_id': edge['pool_id']
                                })
                else:
                    liquidity_sources.append({
                        'type': edge['type'],
                        'data': edge['reserves'],
                        'pool_id': edge['pool_id']
                    })
            elif edge['type'] == 'orderbook':
                liquidity_sources.append({
                    'type': edge['type'],
                    'data': edge['order'],
                    'pool_id': edge['pool_id']
                })
    return liquidity_sources

def add_reverse_edges(all_edges):
    """
    Adds reverse edges to the list of all edges.
    """
    new_edges = []
    for edge in all_edges:
        new_edges.append(edge)
        
        reversed_edge = {
            "from": edge["to"],
            "to": edge["from"],
            "reserves": [edge["reserves"][1], edge["reserves"][0]],
            "pool_id": edge["pool_id"],
            "type": edge.get("type", "amm")
        }
        new_edges.append(reversed_edge)
    return new_edges

def fetch_orderbook(token0, token1):
    """
    Fetches the order book data for a given token pair from the API endpoint.
    """
    url = "https://sphere-router.onrender.com/admin/orderbook"
    payload = {"token0": token0, "token1": token1}
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    formatted_orderbook = {
        "type": "orderbook",
        "from": token0,
        "to": token1,
        "pool_id": "None",
        "order": []
    }

    if token0 == "0xf9529e6c0951efa422d96ca39f8ee582054fd55d":
        if 'data' in data and 'OrderSell' in data['data']:
            for order in data['data']['OrderSell']:
                price = float(order['price'])
                amount = float(order['amount'])
                formatted_orderbook['order'].append([price, amount])
    else:
        if 'data' in data and 'OrderBuy' in data['data']:
            for order in data['data']['OrderBuy']:
                price = float(order['price'])
                amount = float(order['amount'])
                formatted_orderbook['order'].append([amount, price])

    return formatted_orderbook

def calculate_orderbook(orderbook_data, trade_amount):
    """
    Calculates the amount of tokens received based on the order book data and the trade amount.
    """
    total_received = Decimal('0')
    remaining_amount = Decimal(trade_amount)
    print("remaining_amount",remaining_amount)
    for amount_available, price in orderbook_data:
        # print(amount_available,price)
        # exit()
        if remaining_amount <= 0:
            break
        amount_available = Decimal(amount_available)
        price = Decimal(price)
        trade_in_this_order = min(amount_available, remaining_amount)
        received = (trade_in_this_order) * price
        total_received += received
        remaining_amount -= trade_in_this_order
        print(trade_in_this_order , received,total_received,remaining_amount)
    return total_received

@app.route('/optimize_trade', methods=['POST'])
def main():
    """
    Main function to handle the trade optimization request.
    """
    global all_edges
    global edges_orderbook
    start_time = time.time()
    data = request.json
    initial_amount = data.get('initial_amount', 10000)
    start_token = data.get('start_token', "0x9e23efb00426a3d4b51357c048791ab6c3fa5ea0").lower()
    destination_token = data.get('destination_token', "0xf9529e6c0951efa422d96ca39f8ee582054fd55d").lower()
    
    # Fetch all edges (token pairs) from the API
    edges_orderbook = fetch_orderbook(start_token, destination_token)
    all_edges = fetch_edges()
    average_reserves_cache = {}
    
    # Add reverse edges to the list of all edges
    all_edges = add_reverse_edges(all_edges)
    
    # Find all possible paths from the start token to the destination token
    paths = find_paths(start_token=start_token, destination_token=destination_token)
    
    # Format the paths into a graph representation
    edges = format_paths_as_graph(paths)
    print("edges",edges)
    
    # Initialize the node manager
    node_manager = NodeManager()
    
    # Populate the node manager with edges and update the reserves
    for edge in edges:
        token = edge["token"]
        next_token = edge["next_token"]
        pool_ids = edge["pool_ids"]
        types = edge["types"]
        
        for i, reserve_pair in enumerate(edge["reserves"]):
            x, y = reserve_pair
            k = x * y
            edge_type = types[i]
            
            if next_token != "0xf9529e6c0951efa422d96ca39f8ee582054fd55d":
                try:
                    average_reserves = find_average_reserves(next_token, edges, average_reserves_cache)
                except ValueError as e:
                    print(f"Error converting reserves for {token}-{next_token} pair: {e}")
                    # continue
            
            edge_data = {
                'type': edge_type,
                'reserves': [x, y, k],
                'pool_id': pool_ids[i]
            }
            node_manager.add_or_update_node(token, edge_data=edge_data, next_token=next_token)
    
    # Add the order book edge to the node manager if available
    if edges_orderbook:
        add_orderbook_edge(node_manager, edges_orderbook)
    
    # Set the initial amount for the start token
    node_manager.add_or_update_node(start_token, initial_amount)
    
    # Initialize variables for the trade optimization process
    todo = [start_token]
    current_node = node_manager.get_node(start_token)
    token_distribute_result = {}
    token_distribute_node = {}
    
    # Perform trade optimization iteratively
    while todo:
        current_token = todo.pop(0)
        if current_token == destination_token:
            continue
        
        current_node = node_manager.get_node(current_token)
        
        if current_node.amount < MIN_BALANCE_THRESHOLD:
            continue
        
        edges_count = {key: len(value) for key, value in current_node.edges.items()}
        keys = list(edges_count.keys())
        
        # Prepare the liquidity sources data for optimization
        liquidity_sources = prepare_data_for_optimization(current_node, average_reserves_cache)
        print("liquidity_sources",liquidity_sources)
        if len(liquidity_sources) == 0:
            print(f"NO LIQFOUND {current_node.token}",average_reserves_cache,current_node.edges,liquidity_sources)
            return jsonify(average_reserves_cache,current_node.edges,liquidity_sources)
        # Optimize the trade amounts based on the liquidity sources
        distribute_result = optimize_trade(liquidity_sources, current_node.amount)
        print("\nOptimized Trade Amounts:")
        for i, amount in enumerate(distribute_result):
            print(f"Liquidity Source {i+1} (Type: {liquidity_sources[i]['type']}, Pool ID: {liquidity_sources[i].get('pool_id', 'N/A')}): Sell {amount:.4f} A")
        
        # Distribute the optimized trade amounts among the liquidity sources
        for i in range(len(keys)):
            num_elements = edges_count[keys[i]]
            temp_distribute, distribute_result = distribute_result[:num_elements], distribute_result[num_elements:]
            pool_ids_for_key = [pool['pool_id'] for pool in current_node.edges[keys[i]]]
            current_node.distribute_amount(keys[i], temp_distribute, pool_ids_for_key)
            token_distribute_node[current_token] = current_node.distributed
        
        # Update the token balances and reserves based on the trade outcomes
        for next_token, distribute_result in current_node.distributed.items():
            token_received = 0
            for i, (pool_id, amount_to_sell) in enumerate(distribute_result):
                if pool_id and pool_id != 'None':
                    bought_amount = current_node.update_reserves(next_token, pool_id, amount_to_sell)
                    bought_amount = bought_amount * 0.9975
                    token_received += bought_amount
                    if bought_amount is not None:
                        print(f"Sold {amount_to_sell:.4f} of token, bought {bought_amount:.4f} {next_token}")
                        node_manager.add_or_update_node(next_token, amount=bought_amount)
                    else:
                        print("Failed to update reserves. Pool ID or next_token may be incorrect.")
                elif pool_id and pool_id == "None":
                    total_b_received = float(calculate_orderbook(edges_orderbook['order'], amount_to_sell)) * 0.999
                    print(f"Order Book Sold {amount_to_sell:.4f} of token, bought {total_b_received:.4f} {next_token}")
                    token_received += total_b_received
                    node_manager.add_or_update_node(next_token, amount=total_b_received)
            token_distribute_result[next_token] = token_received
        
        # Add the next tokens to the todo list for further processing
        for next_token, amount_received in token_distribute_result.items():
            if next_token != destination_token and amount_received > MIN_BALANCE_THRESHOLD:
                todo.append(next_token)
        
        print("***********xxxxxxxxxxxxxxxxxxxxxxxxxxxx***************")
        print(token_distribute_result)
        end = node_manager.get_node(destination_token)
        print(f"have {end.amount} {destination_token}")
    
    # Print the final trade details
    end = node_manager.get_node(destination_token)
    print(f"We trading {initial_amount} {start_token} for {end.amount} {destination_token}")
    end_time = time.time()
    print("Total Time Taken:", "{:.4f} seconds".format(end_time - start_time))
    
    # Transform the token distribution node to a new format
    token_distribute_ordered = transform_to_new_format(token_distribute_node)
    
    # Prepare the response object
    response = {
        "total_time": end_time - start_time,
        "result_distribute": token_distribute_ordered,
        "result_amount": end.amount
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
