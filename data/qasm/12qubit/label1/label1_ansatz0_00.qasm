OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.5047592972579062) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.5868100884607488) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08748430953718564) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.0308834022815416) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.21543364920208) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2950464918783842) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.01908302044418646) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.008004405597738795) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.39014817296913834) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.21563180819783942) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.24492625529076065) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.04974021294754715) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.12016156398241556) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.12126574358855176) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.20307294882950308) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.13890472421302258) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(3.0058952949777678) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-2.0231495149377077) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(2.642192223378878) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(2.638093165119177) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.9924871008572659) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.21765966163984343) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(2.9240994713501216) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.2017924417096997) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(3.0445995233135252) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(3.042822939426831) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.34225687204645894) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.4730519506140096) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.47552698055365966) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.553500904603908) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.04909674073828402) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.048991637502860924) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(1.741568306914937) q[11];
cx q[10],q[11];
rz(-0.28350133320466714) q[0];
rz(0.641006656760322) q[1];
rz(-1.0012764091360729) q[2];
rz(-1.2871093390918131) q[3];
rz(-0.1885009226768901) q[4];
rz(2.54610696750244) q[5];
rz(-0.0011433096981197633) q[6];
rz(0.5853754146175281) q[7];
rz(-0.7029936136565221) q[8];
rz(-0.4245284818182609) q[9];
rz(0.5660368364265056) q[10];
rz(-1.9849014086622456) q[11];
rx(-2.4484790378955217) q[0];
rx(0.009686049393016205) q[1];
rx(-3.124617168391073) q[2];
rx(3.1415508137191592) q[3];
rx(3.1413723141120498) q[4];
rx(-3.1410135102770034) q[5];
rx(-3.1411778044767606) q[6];
rx(3.1414999115342) q[7];
rx(0.00013908874228060357) q[8];
rx(-0.0009745852895614327) q[9];
rx(3.1406396872211584) q[10];
rx(3.141506601988537) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1885113924219084) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.8787286141066977) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04944565113836215) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.01576885413220597) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.18214261836667817) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.004384497469334263) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.48633767079145046) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.49482503875210526) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.5536000369298053) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.7123482682322472) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.7142686539081318) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(2.342721529181705) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.0135904805783222) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-2.1271949855530625) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.37818027269514803) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.7578594464028808) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-2.382607338890153) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.19982312636437508) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(2.2073116250957354) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.9340557812597087) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.07017710326004661) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(2.2522542096060976) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(2.2517401794911693) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.036320672822287604) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(2.8167187352025715) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(2.8168039083877012) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.16003778248024464) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(2.237851638679199) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(2.2355639952850455) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.49884626968496154) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.2172085814320241) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.21727568142646786) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-1.551140869102658) q[11];
cx q[10],q[11];
rz(-0.5685312795611428) q[0];
rz(-0.6812331799628316) q[1];
rz(1.9038881533787093) q[2];
rz(-1.6384377083573887) q[3];
rz(1.5925753630788184) q[4];
rz(0.645935711924921) q[5];
rz(-0.34130341319826096) q[6];
rz(0.9567689795208454) q[7];
rz(-0.6419246936001074) q[8];
rz(2.661338823502759) q[9];
rz(-0.8198458958480384) q[10];
rz(0.5729108187686517) q[11];
rx(2.9260053898572282) q[0];
rx(2.2794835797607167) q[1];
rx(-3.1395429620403856) q[2];
rx(3.140695308537711) q[3];
rx(-0.00012234155677526873) q[4];
rx(-3.1411700920575254) q[5];
rx(0.00021895094849682517) q[6];
rx(3.1413845898583146) q[7];
rx(3.1410750624828796) q[8];
rx(0.0008753432036941385) q[9];
rx(0.0011493934887711887) q[10];
rx(3.1410345245020275) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.5098121572885255) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-2.131891355553429) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.5855721234694297) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4493975065587517) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.44868977161267165) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(2.5425413079760895) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.31429404694860885) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.314174149348938) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-2.9427265423484674) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.29373537471616074) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.2945492625403857) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.14874619140290316) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(2.789714855531768) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.35016858252135724) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(2.9699607904895595) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.8155826000989499) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.816474424998744) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.4470659985461023) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(2.655366977950873) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-2.656068868642963) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.2591030243935326) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(2.4305263361998004) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(2.430607620035366) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.3511911557980453) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(2.584723117555107) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-2.5843365821471775) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.3770035773067448) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.7390348317404125) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-2.402852995476586) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.46745733516419796) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.6294977990767723) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.6295269007940164) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(2.7049389054092856) q[11];
cx q[10],q[11];
rz(0.4182051290854455) q[0];
rz(2.6649655356782262) q[1];
rz(0.3668679577823444) q[2];
rz(-1.2721126058520411) q[3];
rz(-0.402885172538563) q[4];
rz(-0.7164532450186286) q[5];
rz(-0.47108258715187334) q[6];
rz(-1.7198060053959674) q[7];
rz(1.4517616181325397) q[8];
rz(-1.0897189406398058) q[9];
rz(-0.9072434861753204) q[10];
rz(0.349358981472038) q[11];
rx(-2.2048685355483704) q[0];
rx(-0.005569598543344156) q[1];
rx(-0.0037150565316776962) q[2];
rx(3.140052293929386) q[3];
rx(3.140871018446825) q[4];
rx(3.1415040062721102) q[5];
rx(0.0004246568626640765) q[6];
rx(-3.1409230805094364) q[7];
rx(-3.1415887655193786) q[8];
rx(3.1411314564637673) q[9];
rx(3.1411561535574526) q[10];
rx(3.140800618152628) q[11];