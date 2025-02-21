from bpmn_python.bpmn_diagram_rep import BpmnDiagramGraph

# Create BPMN diagram
bpmn_graph = BpmnDiagramGraph()
bpmn_graph.create_new_diagram_graph(diagram_name="证券经纪业务流程")

# Add pools and lanes
process_id = bpmn_graph.add_process_to_diagram()
pool = bpmn_graph.add_pool_to_diagram("证券经纪业务", process_id)
client_lane = bpmn_graph.add_lane_to_diagram(process_id, "客户", "client_lane")
broker_lane = bpmn_graph.add_lane_to_diagram(process_id, "证券公司", "broker_lane")
exchange_lane = bpmn_graph.add_lane_to_diagram(process_id, "证券交易所", "exchange_lane")
clearing_lane = bpmn_graph.add_lane_to_diagram(process_id, "清算中心", "clearing_lane")

# Add start event
start_id = bpmn_graph.add_start_event_to_diagram(process_id, "start_event")

# Add tasks
open_account = bpmn_graph.add_task_to_diagram(process_id, "开立账户")
verify_docs = bpmn_graph.add_task_to_diagram(process_id, "验证开户材料")
create_accounts = bpmn_graph.add_task_to_diagram(process_id, "创建证券账户和资金账户")

# Add more tasks and gateways
place_order = bpmn_graph.add_task_to_diagram(process_id, "提交交易委托")
verify_order = bpmn_graph.add_task_to_diagram(process_id, "验证委托指令")
match_orders = bpmn_graph.add_task_to_diagram(process_id, "撮合交易")
trading_gateway = bpmn_graph.add_exclusive_gateway_to_diagram(process_id, "是否成交")

# Add sequence flows
bpmn_graph.add_sequence_flow_to_diagram(process_id, start_id, open_account)
bpmn_graph.add_sequence_flow_to_diagram(process_id, open_account, verify_docs)
bpmn_graph.add_sequence_flow_to_diagram(process_id, verify_docs, create_accounts)
bpmn_graph.add_sequence_flow_to_diagram(process_id, create_accounts, place_order)
bpmn_graph.add_sequence_flow_to_diagram(process_id, place_order, verify_order)
bpmn_graph.add_sequence_flow_to_diagram(process_id, verify_order, match_orders)
bpmn_graph.add_sequence_flow_to_diagram(process_id, match_orders, trading_gateway)

# Export diagram
bpmn_graph.export_xml_file("/Users/hongyaotang/src/py/trade/gen/证券经纪业务流程图.bpmn", False)