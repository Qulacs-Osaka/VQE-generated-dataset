OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.0036219830089834204) q[0];
rz(1.2616569626107754) q[0];
ry(2.330041676661058) q[1];
rz(-1.565153666629839) q[1];
ry(3.1415771371882157) q[2];
rz(-1.8794545344909421) q[2];
ry(-0.125019292242729) q[3];
rz(-1.9719516887132462) q[3];
ry(-1.5707203964325658) q[4];
rz(-1.570713334422641) q[4];
ry(-0.3781148423025398) q[5];
rz(-1.9498973849502648) q[5];
ry(1.9015637049622818) q[6];
rz(1.020432352436468) q[6];
ry(-1.5710405704037416) q[7];
rz(-3.038725835576734) q[7];
ry(0.013247141663294671) q[8];
rz(-1.3055166522346093) q[8];
ry(0.8019014300813312) q[9];
rz(2.1457102078919026) q[9];
ry(0.05855235868703878) q[10];
rz(-0.03063487234865114) q[10];
ry(1.7396797256722931) q[11];
rz(-3.1414328951696957) q[11];
ry(-0.0013668078574543498) q[12];
rz(0.846074569035066) q[12];
ry(0.027507093263624327) q[13];
rz(-0.5544245680015107) q[13];
ry(-0.1284624013320903) q[14];
rz(0.7250347700114244) q[14];
ry(1.5692400624922502) q[15];
rz(3.1408393239116847) q[15];
ry(3.1014055109554617) q[16];
rz(1.318914887921502) q[16];
ry(-0.4991447226905857) q[17];
rz(0.011375207598675274) q[17];
ry(0.0012587231242376185) q[18];
rz(-0.6265787339477372) q[18];
ry(-1.5705337061248343) q[19];
rz(1.5741378844412979) q[19];
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
ry(1.570646559276807) q[0];
rz(1.5333693233998664) q[0];
ry(-1.598769622105019) q[1];
rz(-0.10164012455240386) q[1];
ry(-0.07983374333538507) q[2];
rz(-0.9769426077941176) q[2];
ry(-0.0006674002585933047) q[3];
rz(2.3714989780810454) q[3];
ry(1.5707713401797054) q[4];
rz(1.892421685179232) q[4];
ry(-1.075087418405077) q[5];
rz(-2.3730117960719035) q[5];
ry(3.106626164852642) q[6];
rz(0.9156507219535497) q[6];
ry(-3.126745669934768) q[7];
rz(1.6741900972011168) q[7];
ry(-1.570995275448988) q[8];
rz(-1.5702275464260707) q[8];
ry(-2.639874026932887e-06) q[9];
rz(0.8870328475712389) q[9];
ry(-0.7153692618601185) q[10];
rz(-1.574831859603087) q[10];
ry(-1.4279972780835786) q[11];
rz(1.6677188719688572) q[11];
ry(-1.571351737215995) q[12];
rz(0.001136121372274701) q[12];
ry(0.0001579486811280617) q[13];
rz(1.3152550525671893) q[13];
ry(-1.5729933059997059) q[14];
rz(-0.0006082582371771393) q[14];
ry(1.572240226452494) q[15];
rz(-1.6586310223192529) q[15];
ry(-2.871773571260763) q[16];
rz(1.6169407543014347) q[16];
ry(0.018023702736110005) q[17];
rz(-0.18423753446352542) q[17];
ry(-0.905533978187055) q[18];
rz(-0.7361910371042825) q[18];
ry(-0.4989719966331015) q[19];
rz(-0.001833417228570333) q[19];
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
ry(-2.3392370015868593) q[0];
rz(-0.020686941872080356) q[0];
ry(3.114397501752977) q[1];
rz(1.3505647449150084) q[1];
ry(-1.6810119107368848e-05) q[2];
rz(-1.3724003753277942) q[2];
ry(-2.8773814053713016) q[3];
rz(0.01620347094027584) q[3];
ry(2.447111429919624) q[4];
rz(-0.4486075159896789) q[4];
ry(-3.1415473351043555) q[5];
rz(2.8283270247171846) q[5];
ry(1.5309678118115397) q[6];
rz(-2.801887677837486) q[6];
ry(1.5578435835991586) q[7];
rz(-3.1360092442016145) q[7];
ry(-1.5704010926041219) q[8];
rz(3.1360259227208975) q[8];
ry(0.00010062238995583043) q[9];
rz(-2.5827970927183665) q[9];
ry(2.7929163470775347) q[10];
rz(-3.0272250177313347) q[10];
ry(0.04593540811309538) q[11];
rz(1.709126260000306) q[11];
ry(-2.9560368604598644) q[12];
rz(0.000596777714220842) q[12];
ry(-3.903160635854164e-05) q[13];
rz(-1.8491783543120348) q[13];
ry(0.7612267271153547) q[14];
rz(-3.140763359761267) q[14];
ry(-0.0007408506478725485) q[15];
rz(-2.563313757447986) q[15];
ry(-0.1176358357405964) q[16];
rz(-1.5748351906139213) q[16];
ry(2.969666621112174) q[17];
rz(0.8515635395197725) q[17];
ry(3.1413061807250715) q[18];
rz(0.2924449469494678) q[18];
ry(-0.068320997614633) q[19];
rz(-3.139555639136367) q[19];
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
ry(-1.1076845911954503) q[0];
rz(-0.002086828309958052) q[0];
ry(-2.1813471664341737) q[1];
rz(-1.6895704379369922) q[1];
ry(3.037569125476167) q[2];
rz(3.1156190250715685) q[2];
ry(-1.5708975410586592) q[3];
rz(0.001657250175594736) q[3];
ry(-0.00030712070343064113) q[4];
rz(-2.0286365843721565) q[4];
ry(0.11102886675130108) q[5];
rz(3.1155857909830407) q[5];
ry(0.0004975317392334379) q[6];
rz(-2.706148440281139) q[6];
ry(0.028331421837177295) q[7];
rz(-0.005640379096082171) q[7];
ry(-0.5812775352968602) q[8];
rz(-1.2001007619780317) q[8];
ry(-0.010380564025730477) q[9];
rz(2.2778134116396744) q[9];
ry(-0.0002418737480480715) q[10];
rz(-1.8412148805965263) q[10];
ry(-0.00010584172105776446) q[11];
rz(-0.0675051102627151) q[11];
ry(-1.571631665646688) q[12];
rz(1.6861158799325002) q[12];
ry(1.5700370267963037) q[13];
rz(-0.0014282695669170164) q[13];
ry(-1.5728935537859767) q[14];
rz(-1.061091559846381) q[14];
ry(-1.569906043304196) q[15];
rz(3.141196232157006) q[15];
ry(1.8229608141341214) q[16];
rz(0.0050956856396679765) q[16];
ry(3.0556180192402786) q[17];
rz(0.13029171905133707) q[17];
ry(-2.009759592277999) q[18];
rz(-0.28608720123028125) q[18];
ry(-0.43889260757655) q[19];
rz(-0.0030153764355247956) q[19];
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
ry(1.57095114686405) q[0];
rz(-2.5111408000912427) q[0];
ry(-1.2767869715481943) q[1];
rz(-1.6410769235995741) q[1];
ry(0.05098634625222427) q[2];
rz(1.4431420820585976) q[2];
ry(-0.06441743348272855) q[3];
rz(-3.0853299713349993) q[3];
ry(-0.0001972074269618318) q[4];
rz(1.6025444457905582) q[4];
ry(0.23404463840903578) q[5];
rz(1.6153061508854323) q[5];
ry(0.1738507272774097) q[6];
rz(0.7764188225129472) q[6];
ry(-1.5681700707705506) q[7];
rz(0.0005641277506415616) q[7];
ry(-3.141119844016736) q[8];
rz(0.32451342992896764) q[8];
ry(3.141464980793232) q[9];
rz(-2.0538221483673533) q[9];
ry(0.0058016092274773) q[10];
rz(0.924429270600174) q[10];
ry(-0.28933601271596565) q[11];
rz(3.019855455684933) q[11];
ry(4.5000834664245234e-05) q[12];
rz(-0.22252848397893413) q[12];
ry(3.0059159332424397) q[13];
rz(-0.0015337139078509753) q[13];
ry(-3.141579754200037) q[14];
rz(-2.512300201746926) q[14];
ry(3.060239115606623) q[15];
rz(3.141128249909421) q[15];
ry(0.10286394221492934) q[16];
rz(-1.177495453198273) q[16];
ry(3.096073819010757) q[17];
rz(1.6758441350223239) q[17];
ry(0.000654330144013393) q[18];
rz(-0.5400121935171329) q[18];
ry(-1.585611182066453) q[19];
rz(0.11531606034215172) q[19];
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
ry(0.51750221068537) q[0];
rz(-0.6768757666254307) q[0];
ry(3.1355548113620593) q[1];
rz(-1.6045083762496553) q[1];
ry(-1.5713133853524657) q[2];
rz(-1.5544776256140633) q[2];
ry(-0.03807996783241485) q[3];
rz(-0.05746209984189627) q[3];
ry(1.570788530156316) q[4];
rz(2.9480195966924296) q[4];
ry(-0.07688073374429524) q[5];
rz(2.379931501027273) q[5];
ry(-1.570465340883048) q[6];
rz(-3.1409862729151876) q[6];
ry(-1.5998988926776487) q[7];
rz(-3.874921424717083e-05) q[7];
ry(-3.054683525439698) q[8];
rz(-0.045815921603900016) q[8];
ry(-0.03643930721565506) q[9];
rz(3.1206048659698404) q[9];
ry(-0.000183027804326561) q[10];
rz(-0.8577718883863017) q[10];
ry(-3.141466825088755) q[11];
rz(-0.04610178626984299) q[11];
ry(-1.5707710408290831) q[12];
rz(-0.0013565326426103026) q[12];
ry(1.5712304432905997) q[13];
rz(2.319463207568369) q[13];
ry(1.5713998249045404) q[14];
rz(-3.0356720786375013) q[14];
ry(-1.5695523500247104) q[15];
rz(-1.3784596124741986) q[15];
ry(3.0874728204341113) q[16];
rz(-1.1824869829086593) q[16];
ry(0.8789210822027966) q[17];
rz(3.0299957146738006) q[17];
ry(1.6878419393131887) q[18];
rz(0.5968639526472639) q[18];
ry(-0.058764729418763956) q[19];
rz(-1.6859097679633295) q[19];
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
ry(0.1550044863292106) q[0];
rz(-0.5081939838385328) q[0];
ry(-1.57008704656977) q[1];
rz(0.00014545559483902792) q[1];
ry(0.013674343906603922) q[2];
rz(3.1234219081312897) q[2];
ry(-3.0390804772696467) q[3];
rz(3.1393747529138065) q[3];
ry(3.14142026859283) q[4];
rz(1.3764518674462656) q[4];
ry(2.6690477237740964e-05) q[5];
rz(1.6773264725500618) q[5];
ry(-1.5708029344932122) q[6];
rz(0.17456937691759455) q[6];
ry(-1.5659159184528808) q[7];
rz(-3.1410292591887163) q[7];
ry(1.6141097761784198) q[8];
rz(-2.313045188801345) q[8];
ry(3.0373059894102257) q[9];
rz(2.35501169314429) q[9];
ry(2.913865340283492) q[10];
rz(-0.4006600520951258) q[10];
ry(3.103220091820961) q[11];
rz(2.316987797060322) q[11];
ry(0.16436195822842797) q[12];
rz(0.0021967733488787804) q[12];
ry(-3.4529080325640387e-05) q[13];
rz(-1.009151381859229) q[13];
ry(-3.1396078578007742) q[14];
rz(-3.035483916890365) q[14];
ry(-2.4036102965116895e-06) q[15];
rz(0.9887668175329105) q[15];
ry(-0.3534046927720821) q[16];
rz(-1.3530002216476698) q[16];
ry(-3.1036321038252543) q[17];
rz(-0.3319254332840134) q[17];
ry(3.1413475946165983) q[18];
rz(0.22910699368164794) q[18];
ry(1.568033679111462) q[19];
rz(-2.07777655332262) q[19];
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
ry(3.1401114648265405) q[0];
rz(-2.0325307570211297) q[0];
ry(-1.5726806208752526) q[1];
rz(0.03571684839013051) q[1];
ry(-1.5718907305754266) q[2];
rz(-2.9162104221968685) q[2];
ry(-1.570852934842021) q[3];
rz(-1.5968185213856287) q[3];
ry(1.5709502236134518) q[4];
rz(1.8020671367415984) q[4];
ry(-0.0007837689558546929) q[5];
rz(-0.9390501364389261) q[5];
ry(-3.1409317513930954) q[6];
rz(-3.000366256961125) q[6];
ry(1.5709599954606992) q[7];
rz(3.1415741414969647) q[7];
ry(-0.00011253327441712085) q[8];
rz(-0.9290373058216385) q[8];
ry(3.1415164857507247) q[9];
rz(-0.7903894290222047) q[9];
ry(-3.14135714839592) q[10];
rz(-1.98464031608702) q[10];
ry(3.141439302542634) q[11];
rz(0.7447343576749716) q[11];
ry(1.5710534626796682) q[12];
rz(-1.6648802772279847) q[12];
ry(-3.1405544413129634) q[13];
rz(2.955866185594223) q[13];
ry(-1.5703752125796329) q[14];
rz(1.4679621446954307) q[14];
ry(-0.002373271841028827) q[15];
rz(-1.1112081510083918) q[15];
ry(0.00046166514735990205) q[16];
rz(2.799232351013373) q[16];
ry(-3.136541914865407) q[17];
rz(2.938349981181664) q[17];
ry(-3.1410131570228192) q[18];
rz(2.6394675417248314) q[18];
ry(-3.139592802246452) q[19];
rz(1.0815116568924736) q[19];