OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.00025260245630143174) q[0];
rz(0.7047294847737161) q[0];
ry(-0.00019441965708266083) q[1];
rz(2.7954447348305695) q[1];
ry(0.26894847025340873) q[2];
rz(-1.588049428329186) q[2];
ry(1.6734009497683586) q[3];
rz(0.08587376836786743) q[3];
ry(-2.4566071155439344) q[4];
rz(1.3007850749653573) q[4];
ry(-0.00614222374109552) q[5];
rz(0.06763326863521689) q[5];
ry(3.1399921009039273) q[6];
rz(-0.04263716698651621) q[6];
ry(-0.0025473032618501534) q[7];
rz(-1.7260263397189466) q[7];
ry(3.1406629341212495) q[8];
rz(0.1548188083126745) q[8];
ry(3.1180369421827585) q[9];
rz(-0.1816503840771825) q[9];
ry(1.5977756541651775) q[10];
rz(0.007146603834778276) q[10];
ry(1.6410943406104355) q[11];
rz(-3.1234176450370748) q[11];
ry(-3.931716805216839e-05) q[12];
rz(-0.7550997210545898) q[12];
ry(0.052547504157104186) q[13];
rz(-0.04153734432973874) q[13];
ry(-0.0025585319836900575) q[14];
rz(-0.8042521044959191) q[14];
ry(0.0002075836010577703) q[15];
rz(0.8807156062602335) q[15];
ry(1.5699440892148484) q[16];
rz(-2.986269205336843) q[16];
ry(1.5675241364445471) q[17];
rz(-2.5678140628014163) q[17];
ry(3.140394910459093) q[18];
rz(2.5671660014175344) q[18];
ry(-3.1130202479693754) q[19];
rz(0.6698344362419577) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.00023653725525996094) q[0];
rz(-2.911206276126501) q[0];
ry(3.135728267799971) q[1];
rz(0.03902610978348809) q[1];
ry(3.115858684725331) q[2];
rz(-1.4839423465188117) q[2];
ry(-1.103715393307481) q[3];
rz(1.5267558212601247) q[3];
ry(1.2974178230318427) q[4];
rz(3.134079312546296) q[4];
ry(-0.031563040337625026) q[5];
rz(-1.2839105162652311) q[5];
ry(-0.0011363037705682795) q[6];
rz(2.615914829039505) q[6];
ry(-3.1414791700542217) q[7];
rz(-2.043128690299521) q[7];
ry(-2.8672694581771134) q[8];
rz(-0.007275265163772394) q[8];
ry(-1.9083490224691042) q[9];
rz(-3.1331737909603925) q[9];
ry(-1.1991068912311522) q[10];
rz(1.5426245539011578) q[10];
ry(0.3791295149622398) q[11];
rz(-1.5839299703878043) q[11];
ry(1.5717439883232893) q[12];
rz(3.137077592973404) q[12];
ry(1.304602696219014) q[13];
rz(-1.5615953771667561) q[13];
ry(-1.8277569605420814) q[14];
rz(-1.6020569530547384) q[14];
ry(1.5697079759080943) q[15];
rz(-3.140371582346522) q[15];
ry(1.3353059991370924) q[16];
rz(1.4900023312445478) q[16];
ry(-2.8606147949510365) q[17];
rz(-2.584118165805036) q[17];
ry(0.03661390005019438) q[18];
rz(-2.41479141467037) q[18];
ry(-0.3079972525044901) q[19];
rz(-2.653536149542442) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.07934950569105492) q[0];
rz(-0.2765752495519944) q[0];
ry(1.601223336534881) q[1];
rz(2.7921787257298702) q[1];
ry(-3.069619119509923) q[2];
rz(-0.339631895605252) q[2];
ry(0.5184469894498963) q[3];
rz(-1.8542435746937516) q[3];
ry(1.3396337723708207) q[4];
rz(1.6375923204936698) q[4];
ry(-1.5494207797008281) q[5];
rz(-0.01029785019249019) q[5];
ry(-3.112320422073256) q[6];
rz(-2.9386229544069566) q[6];
ry(-1.8744330778526272) q[7];
rz(3.043584307349746) q[7];
ry(-0.12184138665676461) q[8];
rz(-2.1240880833179236) q[8];
ry(-1.680342662109454) q[9];
rz(2.7215044786484204) q[9];
ry(-0.9911005491963346) q[10];
rz(3.010035034306007) q[10];
ry(-0.5821398463463172) q[11];
rz(0.007213179880602993) q[11];
ry(1.567180887256094) q[12];
rz(-0.0005873697726946839) q[12];
ry(1.5981130429890706) q[13];
rz(-3.125468950932395) q[13];
ry(-1.562118021182414) q[14];
rz(1.5893350507228696) q[14];
ry(1.5682560099359293) q[15];
rz(-3.103893561252482) q[15];
ry(-3.1415707957825596) q[16];
rz(-1.777614086941911) q[16];
ry(-4.734895396425791e-05) q[17];
rz(-3.118481759738384) q[17];
ry(-0.22126217313196922) q[18];
rz(-0.03220140039576868) q[18];
ry(1.4811173145320826) q[19];
rz(1.1613474466138483) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1308344530378394) q[0];
rz(-0.06050814513324365) q[0];
ry(-3.137688713898969) q[1];
rz(1.4820296665618908) q[1];
ry(0.00040955124515670063) q[2];
rz(1.0238445092613713) q[2];
ry(0.0004157284042064338) q[3];
rz(2.145532545706142) q[3];
ry(3.0819627495367934) q[4];
rz(-1.289803208307218) q[4];
ry(3.1257005222904817) q[5];
rz(-1.5738712899310014) q[5];
ry(1.6020538902007133) q[6];
rz(-3.1395328210092153) q[6];
ry(1.323529066271777) q[7];
rz(-0.0043944515737281975) q[7];
ry(9.799265169352367e-05) q[8];
rz(0.09872875254175018) q[8];
ry(3.141245821213437) q[9];
rz(2.597694787020962) q[9];
ry(1.8737814525099568) q[10];
rz(2.7108608974147925) q[10];
ry(1.5936699523782893) q[11];
rz(2.6307596097624075) q[11];
ry(-1.6226445494683033) q[12];
rz(2.80633379835106) q[12];
ry(1.5707914517032) q[13];
rz(1.5807419951611326) q[13];
ry(2.10098295072164) q[14];
rz(0.010248597021925043) q[14];
ry(-3.134723536461808) q[15];
rz(-1.5306859300296458) q[15];
ry(0.04103489433982333) q[16];
rz(-2.3176271824399253) q[16];
ry(-0.3013230129208173) q[17];
rz(0.45289843430349364) q[17];
ry(2.7099105768283462) q[18];
rz(-1.0111316883703223) q[18];
ry(-2.2097403308490042) q[19];
rz(2.88036231118781) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.073285479529388) q[0];
rz(-3.0120791012704626) q[0];
ry(-2.9835513499977973) q[1];
rz(0.11829236940967539) q[1];
ry(0.000969445023558002) q[2];
rz(-0.012829921465582927) q[2];
ry(-1.1271802844625256) q[3];
rz(0.5628268026726485) q[3];
ry(3.1411593657575905) q[4];
rz(1.0820348493558578) q[4];
ry(0.9312166578160763) q[5];
rz(-2.8379710830753404) q[5];
ry(-1.6021169385027507) q[6];
rz(2.881974259803476) q[6];
ry(-1.5913000822079184) q[7];
rz(0.7565602194855171) q[7];
ry(3.128333907457135) q[8];
rz(0.12997145440415636) q[8];
ry(3.1399936142411304) q[9];
rz(-0.3634650478155169) q[9];
ry(-3.1409258219713307) q[10];
rz(-0.055041967404749315) q[10];
ry(-0.0029054215225565727) q[11];
rz(3.0994288631461835) q[11];
ry(-0.000595516648402844) q[12];
rz(-0.0885654490220631) q[12];
ry(-3.0812554014695346) q[13];
rz(0.6121911122692059) q[13];
ry(-1.570602491019164) q[14];
rz(-3.0463269748744937) q[14];
ry(-1.8017955887310855) q[15];
rz(2.1731973856782965) q[15];
ry(5.355274667095956e-05) q[16];
rz(2.499766520424428) q[16];
ry(-0.0002651197255980138) q[17];
rz(-0.5672387345369625) q[17];
ry(-0.018266210692011012) q[18];
rz(1.104722485556464) q[18];
ry(-0.3288574197960745) q[19];
rz(2.6219294846587817) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.1402528221812167) q[0];
rz(-2.222465560171619) q[0];
ry(3.0695660563276497) q[1];
rz(-1.7158229822382882) q[1];
ry(-3.060484743175919) q[2];
rz(-2.1618980789483206) q[2];
ry(-3.119948374514068) q[3];
rz(-0.8513400914365752) q[3];
ry(2.760401387160387) q[4];
rz(2.8134897845847333) q[4];
ry(3.125521840784557) q[5];
rz(-2.260210073903393) q[5];
ry(0.2018520273101183) q[6];
rz(0.5155888377985074) q[6];
ry(-1.2200968182722352) q[7];
rz(1.3242891958451564) q[7];
ry(1.5685603216301836) q[8];
rz(-3.13974723582213) q[8];
ry(1.573261111413097) q[9];
rz(3.14093183149635) q[9];
ry(-2.147953063742558) q[10];
rz(-2.1141437059637056) q[10];
ry(-0.011909168647589636) q[11];
rz(0.2713995542605812) q[11];
ry(-3.1415760499412193) q[12];
rz(-0.42371980080727484) q[12];
ry(-0.001322026280161526) q[13];
rz(-2.1730343371255922) q[13];
ry(-3.059948339992279) q[14];
rz(0.04864505928206698) q[14];
ry(-3.141538944388549) q[15];
rz(-1.2584807175855541) q[15];
ry(-2.1780639912112973) q[16];
rz(-2.9692138109131974) q[16];
ry(0.07408407273843136) q[17];
rz(0.3470932119614867) q[17];
ry(-1.1029205963685234) q[18];
rz(3.0713483147915634) q[18];
ry(0.21147545315984576) q[19];
rz(1.6897144645128588) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.585103495298193) q[0];
rz(-1.3686558939999556) q[0];
ry(1.5810191968768867) q[1];
rz(-2.1047096113499437) q[1];
ry(0.001329141181715601) q[2];
rz(-0.4362122124003846) q[2];
ry(-3.1381383011566286) q[3];
rz(-3.0876607353285856) q[3];
ry(0.002760268100923112) q[4];
rz(2.981005195186162) q[4];
ry(-3.131246871706929) q[5];
rz(-1.002103577933184) q[5];
ry(-0.0001560718077348511) q[6];
rz(2.08291363260611) q[6];
ry(-3.141362876704353) q[7];
rz(-0.09658290544629633) q[7];
ry(1.5693159608575722) q[8];
rz(0.005521414352985588) q[8];
ry(1.5727421335186484) q[9];
rz(-0.19949425388366754) q[9];
ry(8.198951312737405e-07) q[10];
rz(-1.0233866290977778) q[10];
ry(-3.1351575069835937) q[11];
rz(-2.6049498102137685) q[11];
ry(-1.571013577376485) q[12];
rz(-3.1364991645425966) q[12];
ry(-1.5709110543851883) q[13];
rz(-0.01545146390538754) q[13];
ry(0.005723475674170634) q[14];
rz(-3.0998095888799635) q[14];
ry(-0.0010979916847399245) q[15];
rz(0.31682144756747993) q[15];
ry(-3.1415916232521166) q[16];
rz(-1.32408213126749) q[16];
ry(3.1415650448231904) q[17];
rz(-0.14397014297762736) q[17];
ry(-2.3712744664904615) q[18];
rz(-1.757546067236519) q[18];
ry(-2.3213273482433188) q[19];
rz(-0.32461875928265554) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1130655651972536) q[0];
rz(-2.9262723220905835) q[0];
ry(1.519497145920545) q[1];
rz(0.1835828853392778) q[1];
ry(-2.894868091189211) q[2];
rz(-0.1536891715966986) q[2];
ry(-0.03195764942296275) q[3];
rz(0.33866867333245976) q[3];
ry(1.2141111427070301) q[4];
rz(0.16698925029156886) q[4];
ry(1.5738246497437163) q[5];
rz(2.6076280417750506) q[5];
ry(2.998372554286671) q[6];
rz(2.1396785828912646) q[6];
ry(-1.2048619931653588) q[7];
rz(-0.14673495314335128) q[7];
ry(1.540132860280952) q[8];
rz(-0.4190966571034027) q[8];
ry(1.6956723009726142) q[9];
rz(-0.9408517829899099) q[9];
ry(-3.1073406844469402) q[10];
rz(0.024335220465972657) q[10];
ry(0.01688252794716916) q[11];
rz(-0.45387857488724737) q[11];
ry(1.5371410142772817) q[12];
rz(-0.5138031321161173) q[12];
ry(1.5689561124707498) q[13];
rz(-1.8513545462273346) q[13];
ry(-1.571676129625958) q[14];
rz(-1.176117119086223) q[14];
ry(-1.714171011755921) q[15];
rz(0.9984043341360717) q[15];
ry(1.7512213211397745) q[16];
rz(-0.9464614790537339) q[16];
ry(-2.580888613436179) q[17];
rz(-0.4043872809332407) q[17];
ry(-2.9669549594772877) q[18];
rz(-0.13088958923921298) q[18];
ry(1.5635184251811036) q[19];
rz(1.9502593968099964) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.7855635923276125) q[0];
rz(-1.0542103989550238) q[0];
ry(-1.5710232515849238) q[1];
rz(-1.8921820313099913) q[1];
ry(-0.021426682894396858) q[2];
rz(-1.4342225330463605) q[2];
ry(3.08029158607003) q[3];
rz(-1.2846872983069169) q[3];
ry(-0.00046928929723843993) q[4];
rz(1.5700300231823139) q[4];
ry(3.141353311985708) q[5];
rz(0.9689721030680909) q[5];
ry(-3.132791532369696) q[6];
rz(-1.7224812801506195) q[6];
ry(-3.1410537630845026) q[7];
rz(-1.6900996737506278) q[7];
ry(0.00945388047672918) q[8];
rz(1.786669907936479) q[8];
ry(-0.014605711240783137) q[9];
rz(1.4811276216653022) q[9];
ry(1.564013533780865) q[10];
rz(-1.5816836311225242) q[10];
ry(1.7902219594545947) q[11];
rz(1.572648104499943) q[11];
ry(0.00022941672390519585) q[12];
rz(-1.0574991804361662) q[12];
ry(0.017164671529537934) q[13];
rz(-1.2977164875655733) q[13];
ry(1.4658695592461825e-06) q[14];
rz(-0.38583612026867464) q[14];
ry(-0.00010661826271516307) q[15];
rz(-1.0012978841213498) q[15];
ry(-0.04140954793202489) q[16];
rz(-1.6767840165882029) q[16];
ry(-3.12891467603037) q[17];
rz(-1.6380073666952715) q[17];
ry(-1.3187579231332205) q[18];
rz(-3.125817359329409) q[18];
ry(3.1413501123373146) q[19];
rz(2.446046592317804) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1277488810036957) q[0];
rz(2.082067273915607) q[0];
ry(3.134238663369875) q[1];
rz(-0.31921440724206374) q[1];
ry(-1.5700642553881523) q[2];
rz(0.0008424254500236657) q[2];
ry(1.570931815175543) q[3];
rz(-3.139780419381639) q[3];
ry(1.2788749824443173) q[4];
rz(1.5833670633141166) q[4];
ry(-1.5706099463486909) q[5];
rz(0.008059303459906351) q[5];
ry(-1.6775733094067213) q[6];
rz(-0.08482388826734467) q[6];
ry(-1.4861486776919364) q[7];
rz(-0.2849205127663481) q[7];
ry(-1.570520301020703) q[8];
rz(0.019689429828634242) q[8];
ry(-1.5704236208393945) q[9];
rz(-0.006097827534880319) q[9];
ry(1.545919157559803) q[10];
rz(-0.01486115489963602) q[10];
ry(-1.5236261374714672) q[11];
rz(1.5696555790417865) q[11];
ry(1.5253035480448587) q[12];
rz(1.5668502936546318) q[12];
ry(-1.571354934760396) q[13];
rz(-3.140975513909215) q[13];
ry(1.5664132809160012) q[14];
rz(0.0007941107133753178) q[14];
ry(1.2923238124300023) q[15];
rz(-1.5434895694328963) q[15];
ry(-2.693019515230269) q[16];
rz(-1.550789969139494) q[16];
ry(2.1396231579260085) q[17];
rz(1.5719237726477215) q[17];
ry(2.716726441982035) q[18];
rz(1.5907343720517737) q[18];
ry(1.567760778323577) q[19];
rz(0.0008240371433273213) q[19];