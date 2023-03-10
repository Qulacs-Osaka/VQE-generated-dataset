OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.00034949815330787004) q[0];
rz(0.5114589273267754) q[0];
ry(-3.1066830221052233) q[1];
rz(-0.7606308875322405) q[1];
ry(0.00807355893603066) q[2];
rz(2.673286395587009) q[2];
ry(0.0431709657968087) q[3];
rz(-0.2295144453873581) q[3];
ry(-0.1829673049552635) q[4];
rz(0.09312850575151922) q[4];
ry(-0.38370664991226816) q[5];
rz(0.023284072675684282) q[5];
ry(-2.597460184565865) q[6];
rz(3.0834550383109423) q[6];
ry(2.6949100178164973) q[7];
rz(0.9851276418765409) q[7];
ry(1.8116735710248404) q[8];
rz(-1.610174558497234) q[8];
ry(-0.10942209469583963) q[9];
rz(2.4491653821762074) q[9];
ry(-3.141568912672153) q[10];
rz(2.2259206095448083) q[10];
ry(-3.1316763713372886) q[11];
rz(-0.9781777766135312) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.141411033124995) q[0];
rz(1.338571065721367) q[0];
ry(-3.1398480626290906) q[1];
rz(1.022882374586123) q[1];
ry(-3.140537993973095) q[2];
rz(0.23047240484694956) q[2];
ry(3.090015970146848) q[3];
rz(1.1173483414801864) q[3];
ry(-3.118256750468228) q[4];
rz(1.1034527226486883) q[4];
ry(0.05372157641669162) q[5];
rz(-1.7329152860486667) q[5];
ry(-0.08678723825996304) q[6];
rz(-1.698273387997678) q[6];
ry(0.018887689153420595) q[7];
rz(-1.5394937138191258) q[7];
ry(1.554153264886516) q[8];
rz(-0.10040577722795216) q[8];
ry(0.2658942222331931) q[9];
rz(2.5086774196816064) q[9];
ry(1.5709977054406723) q[10];
rz(-9.121136095657788e-05) q[10];
ry(-2.9270775167890175) q[11];
rz(-2.1972136712465518) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.894267323436603) q[0];
rz(-1.0826868036923323) q[0];
ry(1.5055986438679625) q[1];
rz(-0.09816860216851797) q[1];
ry(2.901518675987954) q[2];
rz(-1.5131943333087134) q[2];
ry(-3.140876715485097) q[3];
rz(-0.34009776644721806) q[3];
ry(3.1359950253060935) q[4];
rz(-0.5415849539561179) q[4];
ry(3.140140042230425) q[5];
rz(2.9334263074551665) q[5];
ry(0.001346608933875382) q[6];
rz(-2.9593795396670846) q[6];
ry(0.0005350239276252151) q[7];
rz(0.3190384798309056) q[7];
ry(0.0009771468491770463) q[8];
rz(-1.5287969267261978) q[8];
ry(3.141400898806577) q[9];
rz(1.600099196938879) q[9];
ry(-1.570747480240603) q[10];
rz(3.11906179950351) q[10];
ry(-2.7203790126328897e-05) q[11];
rz(-1.1675090543730082) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.103993567697383) q[0];
rz(1.7876530015252396) q[0];
ry(-1.6394475969993743) q[1];
rz(-2.3722585399963223) q[1];
ry(2.8784255480327663) q[2];
rz(-1.0822331509888836) q[2];
ry(1.5537714977960644) q[3];
rz(-3.140621183183662) q[3];
ry(-0.2593169159017566) q[4];
rz(3.1050816879824255) q[4];
ry(0.06750965827891786) q[5];
rz(-0.6026657827057686) q[5];
ry(-0.019217808770288933) q[6];
rz(-1.6614108470993667) q[6];
ry(3.140573649948656) q[7];
rz(-0.13757384159605346) q[7];
ry(-3.137420967637929) q[8];
rz(1.6450873490865137) q[8];
ry(-3.141477291615266) q[9];
rz(-2.6441006673528085) q[9];
ry(3.1396396201568484) q[10];
rz(2.111273348381751) q[10];
ry(-3.141439367508342) q[11];
rz(0.9359115735588139) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.1412186721881277) q[0];
rz(1.2918426948477721) q[0];
ry(-3.1397144172906715) q[1];
rz(-0.7866782294051886) q[1];
ry(0.0023150171805452757) q[2];
rz(1.1486643062129487) q[2];
ry(1.5868142558935716) q[3];
rz(-0.005281916247746839) q[3];
ry(0.0034357299436011957) q[4];
rz(-3.1130696359896453) q[4];
ry(0.003982205884778267) q[5];
rz(-2.499209793340195) q[5];
ry(-0.009007183324000854) q[6];
rz(-1.4471153147706575) q[6];
ry(0.007122891256359643) q[7];
rz(-1.5836949159052347) q[7];
ry(-1.226781308755244) q[8];
rz(-1.409816962379574) q[8];
ry(3.1415468843783705) q[9];
rz(2.2804103806598075) q[9];
ry(2.6930241802797967) q[10];
rz(2.2675738603428783) q[10];
ry(-3.1411539400975417) q[11];
rz(0.331092996010671) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5703969941104656) q[0];
rz(3.141359395630016) q[0];
ry(1.5692770135843865) q[1];
rz(3.139158672392637) q[1];
ry(-1.5732235967219006) q[2];
rz(3.1375825627502048) q[2];
ry(-1.5740473319735202) q[3];
rz(3.102528328730563) q[3];
ry(1.5693755969641732) q[4];
rz(3.0143058313083633) q[4];
ry(1.5723294572665818) q[5];
rz(0.27503866163577406) q[5];
ry(1.594657365423985) q[6];
rz(-0.42717035285416044) q[6];
ry(-1.5495389463999363) q[7];
rz(0.33410798135961034) q[7];
ry(-2.836086426234911) q[8];
rz(-1.6054907207574827) q[8];
ry(-1.5756122382598041) q[9];
rz(-3.0592351873460824) q[9];
ry(-0.7287524200994177) q[10];
rz(1.5319383893152887) q[10];
ry(-1.5685881085018032) q[11];
rz(0.007283347964083475) q[11];