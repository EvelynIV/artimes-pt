结合你现有底层，我建议是这 6 个模块：

grpc_service.py          # gRPC 接口层
command_buffer.py        # 最新目标缓存
control_loop.py          # 100Hz 消费者
control_adapter.py       # 映射/限幅/备压/调用底层
mapping.py               # 弧度映射、零位、方向、减速比
state_store.py           # 保存最新反馈
low_level_controller.py  # 你现有那个双电机流式控制器


gRPC -> 目标命令缓存 -> 100Hz消费者 -> 控制适配层 -> 底层模块
                                        |
                                        -> 映射/限幅/备压/安全处理


控制线程 -> 写 latest_state
gRPC 查询 -> 读 latest_state


客户端
  -> gRPC SetTarget
  -> 写 latest_target
  -> 100Hz 控制线程读取 latest_target
  -> 发给电机
  gRPC handler 是生产者
100Hz 控制线程 是消费者