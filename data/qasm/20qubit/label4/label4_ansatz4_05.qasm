OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-6.794044757896246e-07) q[0];
rz(1.6639642281719178) q[0];
ry(0.032325513446794574) q[1];
rz(0.003652395280972551) q[1];
ry(3.131310978300441) q[2];
rz(0.23751998091828863) q[2];
ry(-7.855534946088483e-06) q[3];
rz(-0.9032695755118405) q[3];
ry(-1.570471929825338) q[4];
rz(-2.1056139566875656) q[4];
ry(1.1020435988581507) q[5];
rz(-2.2636881476792725) q[5];
ry(-1.5707935374153632) q[6];
rz(-1.5707988944186044) q[6];
ry(-1.2759518369756027e-07) q[7];
rz(-0.2746369647593969) q[7];
ry(-3.1414186089506653) q[8];
rz(-0.1685902260323795) q[8];
ry(3.1415919864602286) q[9];
rz(1.5525868838119592) q[9];
ry(-3.1415761625926706) q[10];
rz(-1.4195006058689226) q[10];
ry(-5.488863505913389e-06) q[11];
rz(2.672628598506738) q[11];
ry(-1.5707146193773935) q[12];
rz(0.9816823133565924) q[12];
ry(-1.5707545377507344) q[13];
rz(3.1320013239688262) q[13];
ry(2.947924263142481) q[14];
rz(-0.00021994251137080122) q[14];
ry(1.5717286741693703) q[15];
rz(0.0010268447971203015) q[15];
ry(-0.0006217446641043622) q[16];
rz(-2.8234211793155533) q[16];
ry(1.6527703792914457) q[17];
rz(-1.9505280397219278) q[17];
ry(3.117939943239343) q[18];
rz(-0.004961211841083036) q[18];
ry(-0.00014787718785323945) q[19];
rz(-2.758134491455866) q[19];
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
ry(-1.3277082482332503e-06) q[0];
rz(1.0859246381450696) q[0];
ry(-3.1092702897799964) q[1];
rz(1.6225324256685651) q[1];
ry(1.570816416605318) q[2];
rz(-3.1415919633633487) q[2];
ry(-1.5866617252394744) q[3];
rz(1.8185053673175338) q[3];
ry(-7.072854313783239e-06) q[4];
rz(2.10561155016021) q[4];
ry(-0.000614253073562665) q[5];
rz(-0.8778981783882229) q[5];
ry(-1.5708093443810958) q[6];
rz(-1.544289732425582) q[6];
ry(3.1415762647308063) q[7];
rz(0.5729061106574821) q[7];
ry(-1.570826694843607) q[8];
rz(-0.3893930178965297) q[8];
ry(-1.1732309073496183) q[9];
rz(3.0154049523926627) q[9];
ry(-1.5707672964701427) q[10];
rz(1.814656137705347) q[10];
ry(1.570785965586273) q[11];
rz(1.570751418338598) q[11];
ry(-3.1415904749076504) q[12];
rz(-2.1598960170783927) q[12];
ry(3.114809656489859) q[13];
rz(-0.7673916018628679) q[13];
ry(-0.03676721938455095) q[14];
rz(-0.0008050645030567787) q[14];
ry(0.1930673160291585) q[15];
rz(1.5710197442599383) q[15];
ry(1.6137630549331226) q[16];
rz(3.141579657289876) q[16];
ry(-0.0007106543716899338) q[17];
rz(1.9498130988270805) q[17];
ry(-0.2535877132847677) q[18];
rz(3.1333949553854565) q[18];
ry(-0.000346621928243529) q[19];
rz(-3.010534432112219) q[19];
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
ry(1.5708029178499494) q[0];
rz(0.7473669674154443) q[0];
ry(2.7502593562545026e-06) q[1];
rz(1.681770321388787) q[1];
ry(1.9310391192602365) q[2];
rz(3.1415749328804856) q[2];
ry(0.0024953750460135872) q[3];
rz(1.9985418084721647) q[3];
ry(1.5708009019579918) q[4];
rz(1.9515635610269988) q[4];
ry(-1.57080074651052) q[5];
rz(1.7621223466480114) q[5];
ry(1.570795667258683) q[6];
rz(6.125927119512018e-07) q[6];
ry(3.141591688899339) q[7];
rz(2.5154403585660683) q[7];
ry(-3.141584258301681) q[8];
rz(-1.8279229351685276) q[8];
ry(-1.0314822247003974e-05) q[9];
rz(2.151943804350458) q[9];
ry(-1.5585694057167654) q[10];
rz(-3.0924951802875396) q[10];
ry(-1.571789412171217) q[11];
rz(3.090996929190428) q[11];
ry(-1.5707510821691848) q[12];
rz(1.6051326149680634) q[12];
ry(0.049854123000914706) q[13];
rz(-2.9673393303891586) q[13];
ry(0.13108742725330877) q[14];
rz(-1.5782834536972468) q[14];
ry(-1.5695484575102736) q[15];
rz(-0.12880103175262114) q[15];
ry(-1.5693809470743583) q[16];
rz(1.5706577508128055) q[16];
ry(0.9890842037673021) q[17];
rz(-1.5699905429722625) q[17];
ry(-1.588027997620884) q[18];
rz(3.1399841274747673) q[18];
ry(3.1415129651106177) q[19];
rz(-1.9314451169211067) q[19];
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
ry(3.141585203419287) q[0];
rz(0.725694217865024) q[0];
ry(-2.291255859443938) q[1];
rz(3.1415783790047667) q[1];
ry(-1.5707957738157596) q[2];
rz(-1.570794941468183) q[2];
ry(-9.504669042925684e-07) q[3];
rz(0.8953432213464174) q[3];
ry(-1.5708407700576013) q[4];
rz(-1.5707995001791042) q[4];
ry(-1.5770511107977374) q[5];
rz(-2.7206732090456818) q[5];
ry(1.5710637002111472) q[6];
rz(2.0277456602713073e-07) q[6];
ry(-1.570801532297689) q[7];
rz(1.9266254102857385e-06) q[7];
ry(-3.14158263600606) q[8];
rz(0.13226232423072984) q[8];
ry(-1.1911945728471096e-05) q[9];
rz(2.6867566350971543) q[9];
ry(-1.5707962422286967) q[10];
rz(0.48479497281294165) q[10];
ry(1.5707957185441392) q[11];
rz(3.1415872223523578) q[11];
ry(2.486296179905489) q[12];
rz(1.6452001936573143e-05) q[12];
ry(-8.60252059966183e-05) q[13];
rz(0.5841567154025693) q[13];
ry(3.133334045719284) q[14];
rz(1.5622804483575172) q[14];
ry(3.1415782437755113) q[15];
rz(-0.1288974659257263) q[15];
ry(1.5702417110114508) q[16];
rz(1.6575956863765349) q[16];
ry(1.104093197125188) q[17];
rz(0.5095649512179731) q[17];
ry(-1.5711989599247385) q[18];
rz(-0.4625661242306498) q[18];
ry(3.140658032519007) q[19];
rz(3.1112904258395306) q[19];
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
ry(3.1342835150083954) q[0];
rz(1.5491277821979164) q[0];
ry(1.5708159306459901) q[1];
rz(1.570794318282499) q[1];
ry(1.5708010874232248) q[2];
rz(1.3827238660569632) q[2];
ry(1.5707933174319648) q[3];
rz(9.619196415400212e-06) q[3];
ry(-2.6941512186772436) q[4];
rz(-0.0642997339681397) q[4];
ry(-3.1415687938974046) q[5];
rz(2.2678090344939545) q[5];
ry(1.5707900082247894) q[6];
rz(-2.8105488260223406) q[6];
ry(1.5707955543150582) q[7];
rz(0.9186684234097333) q[7];
ry(1.5707935189393476) q[8];
rz(-0.8170438537679426) q[8];
ry(-2.9095949623737725) q[9];
rz(6.6423788442158274e-06) q[9];
ry(3.141586187901336) q[10];
rz(1.7124030121628093) q[10];
ry(-1.5707779731008307) q[11];
rz(3.048233045176296) q[11];
ry(1.5707317679269839) q[12];
rz(1.5707669972802645) q[12];
ry(1.5707911356709972) q[13];
rz(1.5662634246460527e-05) q[13];
ry(1.570903964734275) q[14];
rz(0.46852550651598845) q[14];
ry(1.5714270835402784) q[15];
rz(-0.268156875157603) q[15];
ry(2.5247252284874415) q[16];
rz(0.05565069528910894) q[16];
ry(-0.6661433442475883) q[17];
rz(0.6654817287514292) q[17];
ry(3.1411420290092162) q[18];
rz(2.5178787908864497) q[18];
ry(-1.2871214326511342) q[19];
rz(-2.0559937960615295) q[19];
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
ry(1.5707909766497572) q[0];
rz(-1.5707960197215307) q[0];
ry(-1.5707949229967004) q[1];
rz(-2.579435259519869e-05) q[1];
ry(-3.1415918341948683) q[2];
rz(-1.4385702434753236) q[2];
ry(1.5862500324136048) q[3];
rz(-0.4294247847167376) q[3];
ry(0.0090032561472011) q[4];
rz(-3.077282162954559) q[4];
ry(-3.135263773801359) q[5];
rz(2.1835446781831243) q[5];
ry(3.1415920690572707) q[6];
rz(1.9018397397910498) q[6];
ry(5.053678080147701e-07) q[7];
rz(-2.489464991425891) q[7];
ry(-3.1415913189002875) q[8];
rz(0.9224644260339103) q[8];
ry(2.8315322076558216) q[9];
rz(-0.0001948623466816457) q[9];
ry(0.00026405587837218284) q[10];
rz(2.831798623318101) q[10];
ry(-6.80990296396308e-05) q[11];
rz(1.656205117863852) q[11];
ry(1.1325507321980295) q[12];
rz(-2.7760404386180584) q[12];
ry(-1.5708192239151673) q[13];
rz(1.5707903604830804) q[13];
ry(-3.1415810356237532) q[14];
rz(-0.3030493453033917) q[14];
ry(3.1415881331682325) q[15];
rz(2.873294894940733) q[15];
ry(-3.1414560207706463) q[16];
rz(1.5545194971435246) q[16];
ry(0.0007025193783825046) q[17];
rz(1.8598716569598261) q[17];
ry(0.0006665246595369823) q[18];
rz(2.281590042775872) q[18];
ry(-3.1376976309079203) q[19];
rz(1.6173975472330469) q[19];
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
ry(1.5707345864301223) q[0];
rz(1.8612107337525852) q[0];
ry(-1.5705639515660748) q[1];
rz(0.30678130479696547) q[1];
ry(-4.423827709054023e-06) q[2];
rz(-1.8910710979286875) q[2];
ry(3.6526237713552234e-05) q[3];
rz(0.430425302141268) q[3];
ry(2.7017403803420272) q[4];
rz(0.09334766109256697) q[4];
ry(3.141574610117908) q[5];
rz(-1.4317011719359742) q[5];
ry(-1.5707949260677194) q[6];
rz(-2.933153617140731) q[6];
ry(-1.5707978647704781) q[7];
rz(1.933418578840258) q[7];
ry(-1.030378447001809e-06) q[8];
rz(2.972878435518799) q[8];
ry(-1.8027931591121682) q[9];
rz(3.141547765702209) q[9];
ry(-2.291726990148265e-05) q[10];
rz(-0.9178127739595118) q[10];
ry(1.5707915779695591) q[11];
rz(1.5708144862265092) q[11];
ry(3.1415893564329087) q[12];
rz(0.18123978908252344) q[12];
ry(-1.5708693411799581) q[13];
rz(2.7834660488257152) q[13];
ry(1.8407440235865007e-05) q[14];
rz(2.4745501805793655) q[14];
ry(1.5708464116105405) q[15];
rz(1.5098952453259613) q[15];
ry(1.519725250833656) q[16];
rz(0.9558603218560666) q[16];
ry(1.0005189174657545) q[17];
rz(2.0031527562493623) q[17];
ry(-3.141413059543405) q[18];
rz(-2.427499047863913) q[18];
ry(2.8157428646202796) q[19];
rz(-2.6326987949448832) q[19];
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
ry(9.50081801054381e-06) q[0];
rz(2.851392087689217) q[0];
ry(-5.947265785864543e-06) q[1];
rz(2.9513259733271644) q[1];
ry(-1.5708065486907663) q[2];
rz(0.5137114257387196) q[2];
ry(3.1267888089367366) q[3];
rz(1.515458812537942) q[3];
ry(3.141592246220212) q[4];
rz(0.09331305436468439) q[4];
ry(2.8615113745189547e-07) q[5];
rz(1.6579047742572117) q[5];
ry(-3.1415915894892734) q[6];
rz(-1.3623578811304908) q[6];
ry(-2.7543737868285234e-07) q[7];
rz(1.2081745692359716) q[7];
ry(1.5707983707668989) q[8];
rz(1.5708026671133402) q[8];
ry(-1.269627713532433) q[9];
rz(1.570953057975335) q[9];
ry(1.5705214621093326) q[10];
rz(0.43195212412316586) q[10];
ry(-1.5708017509527987) q[11];
rz(3.141482415598259) q[11];
ry(-3.141586551286016) q[12];
rz(1.3862881804818212) q[12];
ry(3.1415900392863123) q[13];
rz(-1.9289153206064968) q[13];
ry(0.00027371528981992604) q[14];
rz(3.0093838469089587) q[14];
ry(3.1415600224941427) q[15];
rz(2.914714755090041) q[15];
ry(-2.6312535821908596) q[16];
rz(1.5708103570010756) q[16];
ry(3.141584159573689) q[17];
rz(-0.06437889547489828) q[17];
ry(0.0003319810433550785) q[18];
rz(-1.7350008138384454) q[18];
ry(-1.571236495656805) q[19];
rz(6.684029398584812e-05) q[19];
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
ry(1.5707605936168065) q[0];
rz(-2.184909302628162) q[0];
ry(-1.5707989369412418) q[1];
rz(-2.809877829527264) q[1];
ry(-3.1415261539854455) q[2];
rz(-1.6711986421195109) q[2];
ry(-3.006343265585332e-06) q[3];
rz(-1.1827432254714316) q[3];
ry(1.5707915360342557) q[4];
rz(-0.6141121467837243) q[4];
ry(-1.5707963700163607) q[5];
rz(-2.8098722673104444) q[5];
ry(1.570802773131498) q[6];
rz(-0.6141114707333912) q[6];
ry(1.5707810858552889) q[7];
rz(0.33171967541914016) q[7];
ry(1.5707911593056008) q[8];
rz(-0.6141035576377235) q[8];
ry(-0.12153474458152756) q[9];
rz(1.9023583520379712) q[9];
ry(3.141588882534784) q[10];
rz(-1.7529349333286601) q[10];
ry(-1.5707950727129507) q[11];
rz(1.9025156652689594) q[11];
ry(-1.5707841788416057) q[12];
rz(0.9567711799036536) q[12];
ry(-1.3765662362346793) q[13];
rz(1.902516653878772) q[13];
ry(-1.5707893940979383) q[14];
rz(-2.1847568113806366) q[14];
ry(-0.00016668132477448694) q[15];
rz(-2.6438940112754308) q[15];
ry(-1.5696548785279734) q[16];
rz(-0.6148900814093018) q[16];
ry(1.5707206185939189) q[17];
rz(0.33194026678626815) q[17];
ry(-1.5704849406582442) q[18];
rz(-2.1849619990664655) q[18];
ry(-1.570616028950826) q[19];
rz(-1.239049351184904) q[19];