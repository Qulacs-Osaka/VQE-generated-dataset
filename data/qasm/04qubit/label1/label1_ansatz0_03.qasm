OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.3453090394342537) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.2764280062849052) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09589593373034899) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.2233128060644138) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.406029392022842) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19788685495433497) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.1547217647028047) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.7833165384211336) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10381149871199263) q[3];
cx q[2],q[3];
rz(-0.2570425945467436) q[0];
rz(0.08525575973608994) q[1];
rz(0.017863807886486005) q[2];
rz(-0.08130504843992727) q[3];
rx(-1.668414442093791) q[0];
rx(-0.8618213334748684) q[1];
rx(-1.4046205204328648) q[2];
rx(-1.4587184999876626) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.2178952761221284) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.2054110012576622) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2240382774119751) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.8109892598924648) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3018007483182293) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3822204305668829) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.646632584569132) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.1344441496592512) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.030492467482727513) q[3];
cx q[2],q[3];
rz(-0.15765109242182748) q[0];
rz(0.3608943403595761) q[1];
rz(-0.019143244218950382) q[2];
rz(-0.2656096250154432) q[3];
rx(-1.3992880168410782) q[0];
rx(-0.7967009504348463) q[1];
rx(-1.5114875098275153) q[2];
rx(-1.6146582626771018) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.233148596400574) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.90505970830167) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03977792054536492) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9376328201711239) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.7032207042409016) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.4857069424695423) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.551769670230633) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.8969493727227985) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.41904840272670996) q[3];
cx q[2],q[3];
rz(-0.13707039983658975) q[0];
rz(-0.032669887306300945) q[1];
rz(-0.0832924136302804) q[2];
rz(0.007549061366360043) q[3];
rx(-1.9220269032105752) q[0];
rx(-0.3431314231582335) q[1];
rx(-1.3021998884307475) q[2];
rx(-1.2456970563078822) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.4304852591387436) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.4375542727970503) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.10277160300263277) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.513356777593237) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.1519797177561342) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3002902603575265) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.9073026171758192) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.0928267084586885) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.015853292250182716) q[3];
cx q[2],q[3];
rz(0.5739750360639978) q[0];
rz(0.32130900821942593) q[1];
rz(-0.24153396159747623) q[2];
rz(-0.1707014226278006) q[3];
rx(-2.231532708595847) q[0];
rx(-0.6042877904008706) q[1];
rx(-1.207531456528277) q[2];
rx(-0.880690235167972) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.4262252432396405) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.072045369773429) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.6739081559693068) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.6790197799932878) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.9893376546377397) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.23156631021229235) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5908139030751695) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6986399944185574) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1768576002738441) q[3];
cx q[2],q[3];
rz(-0.3343834700942706) q[0];
rz(0.2144066747611388) q[1];
rz(0.06381599738544295) q[2];
rz(0.01153420514642983) q[3];
rx(-2.685634294881268) q[0];
rx(-0.4897415951382405) q[1];
rx(-1.0176671353190954) q[2];
rx(-1.0451504213327143) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.443126140585323) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.6027264115079358) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.019528282720629588) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9218089917045134) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.0500828080927673) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.46182808733236796) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0021548602258988836) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.9578752147058627) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.20532786926236568) q[3];
cx q[2],q[3];
rz(-0.3971790332503252) q[0];
rz(0.1969030262151353) q[1];
rz(-0.2790615176308292) q[2];
rz(-0.11768339254112488) q[3];
rx(-1.3258314579010686) q[0];
rx(-0.8634274086137879) q[1];
rx(-1.5002859912864106) q[2];
rx(-1.2143117971866388) q[3];