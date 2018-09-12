# 训练数据
## 训练数据字段及其含义
| 字段 | 中文名 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| USERID | 用户ID | VARCHAR2(50) | 用户编码，标识用户的唯一字段 | 
| current_type | 套餐 | VARCHAR2(500) | / |
| service_type|套餐类型  | VARCHAR2(10) | 0：23G融合，1：2I2C，2：2G，3：3G，4：4G |
| is_mix_service | 是否固移融合套餐 | VARCHAR2(10) | 1.是 0.否 |
| online_time | 在网时长 | VARCHAR2(50) | / |
| 1_total_fee |当月总出账金额_月 | NUMBER | 单位：元 |
| 2_total_fee |当月前1月总出账金额_月 | NUMBER | 单位：元
| 3_total_fee |当月前2月总出账金额_月 | NUMBER | 单位：元 |
| 4_total_fee |当月前3月总出账金额_月 | NUMBER | 单位：元 |
| month_traffic | 当月累计-流量 | NUMBER | 单位：MB |
| many_over_bill | 连续超套 | VARCHAR2(500) | 1-是，0-否 |
| contract_type | 合约类型 | VARCHAR2(500) | ZBG_DIM.DIM_CBSS_ACTIVITY_TYPE |
| contract_time | 合约时长 | VARCHAR2(500) | / |
| is_promise_low_consume | 是否承诺低消用户 | VARCHAR2(500) | 1.是 0.否 |
| net_service |网络口径用户 | VARCHAR2(500) | 20AAAAAA-2G |
| pay_times | 交费次数 | NUMBER | 单位：次 |
| pay_num | 交费金额 | NUMBER | 单位：元 |
| last_month_traffic | 上月结转流量 | NUMBER |单位：MB |
| local_trafffic_month | 月累计-本地数据流量 | NUMBER | 单位：MB |
| local_caller_time | 本地语音主叫通话时长 | NUMBER | 单位：分钟 |
| service1_caller_time | 套外主叫通话时长 | NUMBER | 单位：分钟 |
| service2_caller_time | Service2_caller_time | NUMBER | 单位：分钟 |
| gender | 性别 | varchar2(100) | 01.男 02女 |
| age | 年龄 | varchar2(100) | / |
| complaint_level | 投诉重要性 | VARCHAR2(1000) | 1：普通，2：重要，3：重大 |
| former_complaint_num | 交费金历史投诉总量 | NUMBER | 单位：次 |
| former_complaint_fee | 历史执行补救费用交费金额 | NUMBER | 单位：分 |

## 数据类型
```angular2html
# 用户ID
user_id                    object

# 离散数据
service_type                int64
is_mix_service              int64
many_over_bill              int64
contract_type               int64
is_promise_low_consume      int64
net_service                 int64
gender                      int64
complaint_level             int64
age                         int64

# 连续数据
online_time                 int64
1_total_fee               float64
2_total_fee               float64
3_total_fee               float64
4_total_fee               float64
month_traffic             float64
contract_time               int64
pay_times                   int64
pay_num                   float64
last_month_traffic        float64
local_trafffic_month      float64
local_caller_time         float64
service1_caller_time      float64
service2_caller_time      float64
former_complaint_num        int64
former_complaint_fee      float64

# 标签
current_service             int64
```
```
"service_type",
"is_mix_service",
"many_over_bill",
"contract_type",
"is_promise_low_consume",
"net_service",
"gender",
"complaint_level",
"age"

# 连续数据
"online_time",
"1_total_fee",
"2_total_fee",
"3_total_fee",
"4_total_fee",
"month_traffic", 
"contract_time",
"pay_times",
"pay_num",
"last_month_traffic",
"local_trafffic_month",
"local_caller_time",
"service1_caller_time",
"service2_caller_time",
"former_complaint_num",
"former_complaint_fee"
```