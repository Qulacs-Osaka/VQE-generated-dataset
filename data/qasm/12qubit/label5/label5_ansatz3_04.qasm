OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.010021784314972679) q[0];
rz(-2.1072823779235614) q[0];
ry(1.3123665872578547) q[1];
rz(0.0036820952451927964) q[1];
ry(3.1313262689212777) q[2];
rz(-1.4606148498514164) q[2];
ry(2.50179436869509) q[3];
rz(-3.055801751205725) q[3];
ry(2.4102467823980063) q[4];
rz(2.7963723696078753) q[4];
ry(0.0001900304982216383) q[5];
rz(-2.648361799289792) q[5];
ry(1.8973604586362889) q[6];
rz(-1.8031454192937983) q[6];
ry(1.6026619520835181) q[7];
rz(-0.8637087009622038) q[7];
ry(-1.5302480151546156) q[8];
rz(1.5724296064119763) q[8];
ry(1.5762700654130064) q[9];
rz(-3.139212012479813) q[9];
ry(-1.5701133067293356) q[10];
rz(0.004308657673388616) q[10];
ry(-1.5672879976845078) q[11];
rz(0.847147170199586) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.560540944253426) q[0];
rz(-1.5446877807315067) q[0];
ry(1.2518115353213237) q[1];
rz(-2.9061865562626243) q[1];
ry(-1.5727691825201329) q[2];
rz(-1.3808306654734879) q[2];
ry(0.21610000090068393) q[3];
rz(-1.212725370788175) q[3];
ry(-2.788482491071875) q[4];
rz(-1.8908832759146301) q[4];
ry(-1.5486140992982065) q[5];
rz(-2.153986940348698) q[5];
ry(-0.008876133492074164) q[6];
rz(-2.1825918632678185) q[6];
ry(0.0358889339938286) q[7];
rz(-2.2853685201142606) q[7];
ry(1.5842175103152265) q[8];
rz(1.572159531160695) q[8];
ry(-1.5658843459820637) q[9];
rz(2.6852406316456623) q[9];
ry(1.5693806049181265) q[10];
rz(1.5132939581992169) q[10];
ry(1.4915204235381374) q[11];
rz(-1.9352835328711533) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.6374352141700683) q[0];
rz(1.8275866273772223) q[0];
ry(0.006749152046003051) q[1];
rz(0.5250110022108503) q[1];
ry(0.530702741283215) q[2];
rz(2.723105203934298) q[2];
ry(0.9857661087525946) q[3];
rz(-0.8881564415657879) q[3];
ry(-0.0022325621322867306) q[4];
rz(-3.1314324289553315) q[4];
ry(-3.1404377253400835) q[5];
rz(0.9874192680683447) q[5];
ry(-3.110334510962152) q[6];
rz(-2.4259437465740294) q[6];
ry(-2.5300605152404056) q[7];
rz(-3.136256748659102) q[7];
ry(-1.5719566490349193) q[8];
rz(-1.5746769785371453) q[8];
ry(1.0990348327739747) q[9];
rz(0.830877347409899) q[9];
ry(1.5721962214681107) q[10];
rz(1.5766417058818663) q[10];
ry(-0.020226314420398447) q[11];
rz(-3.0227201987464722) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5351570281894416) q[0];
rz(-1.0019883944571433) q[0];
ry(-1.5631648995468554) q[1];
rz(-1.2233183547667963) q[1];
ry(0.005405571756616823) q[2];
rz(-2.692516673570245) q[2];
ry(0.00044066739278658673) q[3];
rz(2.995338339526949) q[3];
ry(1.4835400900456444) q[4];
rz(-2.6574140055238953) q[4];
ry(1.5531074441630324) q[5];
rz(-2.4300987037513924) q[5];
ry(-1.600769484100629) q[6];
rz(-3.1318987037085204) q[6];
ry(1.4896947319414844) q[7];
rz(-1.8064210002885535) q[7];
ry(-1.5694170464545425) q[8];
rz(-1.4718486291702084) q[8];
ry(0.9543302287594384) q[9];
rz(-3.11912702667182) q[9];
ry(-1.5798661293379879) q[10];
rz(0.2627890779180963) q[10];
ry(-0.004179657341615992) q[11];
rz(-3.111109627622006) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.1351849666621336) q[0];
rz(-2.6070396462471583) q[0];
ry(0.005430037249992914) q[1];
rz(2.786166940844608) q[1];
ry(1.7526525679777194) q[2];
rz(1.8383880302313806) q[2];
ry(1.8747644369270924) q[3];
rz(-0.4394194777216442) q[3];
ry(-1.5697143159277491) q[4];
rz(-0.9098768882128052) q[4];
ry(3.1414785692645757) q[5];
rz(2.8582996260854308) q[5];
ry(1.2269323746580134) q[6];
rz(1.588087230623005) q[6];
ry(0.0036586866561096536) q[7];
rz(2.900307626583988) q[7];
ry(-0.013768353533716747) q[8];
rz(0.39918031542689264) q[8];
ry(-1.1178685602073033) q[9];
rz(-0.24003553465392308) q[9];
ry(-0.0073622549204598355) q[10];
rz(-0.11946815965755421) q[10];
ry(1.6227471830343347) q[11];
rz(1.5571568780245875) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.0257139566196232) q[0];
rz(1.7364215422093165) q[0];
ry(-2.1614498466314305) q[1];
rz(-0.18375528315728393) q[1];
ry(-0.08853133352646514) q[2];
rz(1.8889450170098305) q[2];
ry(-0.007082047432721338) q[3];
rz(-1.1740603292583405) q[3];
ry(1.5724581424837725) q[4];
rz(-0.3487907045145251) q[4];
ry(-3.1275812371509386) q[5];
rz(-2.5438802350944036) q[5];
ry(1.7466652575112862) q[6];
rz(-0.020177955240566873) q[6];
ry(-3.139769130945668) q[7];
rz(0.9288421646877963) q[7];
ry(-0.0004880483981004966) q[8];
rz(2.092226441179963) q[8];
ry(-0.06705149658968346) q[9];
rz(1.8052137688515717) q[9];
ry(-1.5351442333613623) q[10];
rz(1.5608647134950262) q[10];
ry(1.574219680111918) q[11];
rz(-2.510545793135112) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.128537448701942) q[0];
rz(0.11825306315500615) q[0];
ry(-3.133127123200643) q[1];
rz(1.398500948430243) q[1];
ry(-1.5272433193288686) q[2];
rz(1.7260198552750312) q[2];
ry(1.952731375408982) q[3];
rz(0.24252660905013615) q[3];
ry(-3.1312554311685936) q[4];
rz(-0.34989086388234686) q[4];
ry(-0.0020834015512119367) q[5];
rz(-0.04913504964773582) q[5];
ry(0.04277668700618342) q[6];
rz(1.5973488267140319) q[6];
ry(3.137501437736313) q[7];
rz(0.5279999930280254) q[7];
ry(3.0461557610413386) q[8];
rz(1.1117711882779133) q[8];
ry(-1.579316797658338) q[9];
rz(2.1144559335989737) q[9];
ry(-1.571909482083392) q[10];
rz(-1.5841925060961053) q[10];
ry(3.1202329406374836) q[11];
rz(0.6362787195718552) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.5677975433174227) q[0];
rz(-3.062882856219676) q[0];
ry(1.5530735644459908) q[1];
rz(0.805518250638996) q[1];
ry(-3.1329341433762354) q[2];
rz(0.19703774002373287) q[2];
ry(3.139531396712878) q[3];
rz(-0.46470825522353115) q[3];
ry(1.1879283356996142) q[4];
rz(-3.0815210606430385) q[4];
ry(-3.1339024690800272) q[5];
rz(-0.37435771913786026) q[5];
ry(-1.5408442035958085) q[6];
rz(-1.7208439307054437) q[6];
ry(-3.1380284683955977) q[7];
rz(1.8795822180831414) q[7];
ry(1.5787096205110913) q[8];
rz(1.609215189997698) q[8];
ry(-3.134490552624248) q[9];
rz(-1.3903614740399757) q[9];
ry(-1.692046657495395) q[10];
rz(-2.9669036805682585) q[10];
ry(-1.576305426248096) q[11];
rz(-0.3354320490577001) q[11];