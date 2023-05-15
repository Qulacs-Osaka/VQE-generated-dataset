OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.9421013730858361) q[0];
rz(-1.2561307566846704) q[0];
ry(-0.0031514441812410254) q[1];
rz(-0.793721510552994) q[1];
ry(-2.6215435310830797) q[2];
rz(-0.28996243757049983) q[2];
ry(1.5693486971583777) q[3];
rz(-1.5642591253957359) q[3];
ry(-0.004352923787389593) q[4];
rz(1.2108975247149791) q[4];
ry(3.1415071257760148) q[5];
rz(2.838584384018204) q[5];
ry(-1.245643230095646) q[6];
rz(0.026241298552031722) q[6];
ry(3.138248786955981) q[7];
rz(1.3409646147656908) q[7];
ry(1.5823454583399883) q[8];
rz(0.9820752365004869) q[8];
ry(0.007555837555705658) q[9];
rz(-1.3923583639143622) q[9];
ry(-1.0804189351878286) q[10];
rz(3.026120206620633) q[10];
ry(-0.00023008830589610343) q[11];
rz(0.08027518926848476) q[11];
ry(1.0290772557216787) q[12];
rz(0.015119909563332532) q[12];
ry(3.0898225838382514) q[13];
rz(2.750910397800495) q[13];
ry(0.020180568977863602) q[14];
rz(0.20453241499412655) q[14];
ry(1.9271241391965495) q[15];
rz(2.9922343912491782) q[15];
ry(0.012435515772062365) q[16];
rz(-1.794358898849354) q[16];
ry(3.1396821906596255) q[17];
rz(2.521542418558051) q[17];
ry(-1.0057793988248154) q[18];
rz(-1.710844226269638) q[18];
ry(0.9626056608965632) q[19];
rz(2.5560932807533807) q[19];
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
ry(0.003110876536536189) q[0];
rz(2.3952645500192595) q[0];
ry(-0.27579127597643893) q[1];
rz(-3.007041586775108) q[1];
ry(3.1323166468358483) q[2];
rz(-1.4720241994239387) q[2];
ry(-1.181754407766341) q[3];
rz(-0.44995220342987424) q[3];
ry(-0.006315635525278204) q[4];
rz(-1.7443096595060643) q[4];
ry(3.024619843868067) q[5];
rz(0.8054210754150192) q[5];
ry(-1.61645196780571) q[6];
rz(-2.2296592500566876) q[6];
ry(-0.0022654391779646232) q[7];
rz(-1.1566279162027695) q[7];
ry(0.038661355119485374) q[8];
rz(0.1738384260943038) q[8];
ry(3.138575108937952) q[9];
rz(2.9870564695319914) q[9];
ry(1.5620384348469079) q[10];
rz(-0.2735102506145566) q[10];
ry(-3.141570804443622) q[11];
rz(-2.4042102731080335) q[11];
ry(-1.7665118858734887) q[12];
rz(-1.064249153312809) q[12];
ry(1.1355896692779734) q[13];
rz(-2.622203734249005) q[13];
ry(-0.11007058511492845) q[14];
rz(1.6248273821044519) q[14];
ry(2.5928174890610425) q[15];
rz(0.6505595542154357) q[15];
ry(3.1324334950309565) q[16];
rz(2.089840092600845) q[16];
ry(-3.103823095973874) q[17];
rz(0.4687687834133376) q[17];
ry(0.2774569110330152) q[18];
rz(1.176747087243046) q[18];
ry(0.753996795310031) q[19];
rz(-3.127667154235424) q[19];
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
ry(1.7020791549449399) q[0];
rz(-1.979999690787495) q[0];
ry(-0.1998015135636626) q[1];
rz(-2.7472860819307967) q[1];
ry(2.62297835995212) q[2];
rz(1.6254118570168894) q[2];
ry(0.6530493345128026) q[3];
rz(1.8958509177944292) q[3];
ry(0.0026107565893249865) q[4];
rz(1.3346771279897274) q[4];
ry(-3.0335704647941664) q[5];
rz(0.15193147897022122) q[5];
ry(-1.1915464805664808) q[6];
rz(0.44266840885533476) q[6];
ry(-3.1278104528508703) q[7];
rz(-1.10388533030107) q[7];
ry(1.144814379566779) q[8];
rz(-2.11393902640257) q[8];
ry(3.1353978177530664) q[9];
rz(-0.06401297988717757) q[9];
ry(1.3734526980726385) q[10];
rz(-0.4358527726819617) q[10];
ry(0.00030220562504723597) q[11];
rz(-0.5891234707548288) q[11];
ry(2.2541130742692888) q[12];
rz(-0.11200111519702098) q[12];
ry(-0.4823759986451048) q[13];
rz(3.0375411435205875) q[13];
ry(-1.7965953654055045) q[14];
rz(-2.634127598867973) q[14];
ry(-0.03143334920936792) q[15];
rz(-0.7827084062842079) q[15];
ry(-3.118381478485972) q[16];
rz(3.095853077858898) q[16];
ry(-0.00130785520783791) q[17];
rz(0.15453296278489237) q[17];
ry(1.6240267231871415) q[18];
rz(-0.5797469358812777) q[18];
ry(-0.07766479049278896) q[19];
rz(-2.905743566445387) q[19];
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
ry(-0.42895235151231287) q[0];
rz(0.32930512707224446) q[0];
ry(1.5441057375324478) q[1];
rz(2.376157969380674) q[1];
ry(0.00900737651093811) q[2];
rz(-0.004717555064315704) q[2];
ry(2.098536818292327) q[3];
rz(-1.2293269908459366) q[3];
ry(-1.2831118104692445) q[4];
rz(3.0367453410199445) q[4];
ry(-3.0716691276749404) q[5];
rz(2.1726294763922422) q[5];
ry(0.1696111568798191) q[6];
rz(0.39812081833186824) q[6];
ry(-0.00970586154558184) q[7];
rz(1.0720268585064285) q[7];
ry(3.066578985943929) q[8];
rz(-2.198538090426419) q[8];
ry(-3.1326757284657516) q[9];
rz(1.424626651002078) q[9];
ry(-0.0005238352627108256) q[10];
rz(-0.2830461597706726) q[10];
ry(3.141364481677205) q[11];
rz(0.16667642331432386) q[11];
ry(3.072036296610109) q[12];
rz(3.0685943303700878) q[12];
ry(-0.02982329111592932) q[13];
rz(1.607852771797275) q[13];
ry(-0.006469875839028917) q[14];
rz(-0.572653746413545) q[14];
ry(3.1330121398012007) q[15];
rz(3.0501049877584863) q[15];
ry(-2.820755805506427) q[16];
rz(-0.5827891236071487) q[16];
ry(0.07042792436812827) q[17];
rz(2.331242202523569) q[17];
ry(0.7657858010955378) q[18];
rz(-1.6299805861072807) q[18];
ry(1.1578776073385892) q[19];
rz(-0.5989008823995688) q[19];
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
ry(0.6189895669986638) q[0];
rz(1.7259566378054974) q[0];
ry(2.566936864444094) q[1];
rz(0.6326236851570046) q[1];
ry(-3.11588306840765) q[2];
rz(0.9795732204767349) q[2];
ry(1.171394006302528) q[3];
rz(-1.4242391286521627) q[3];
ry(-0.01637855236760091) q[4];
rz(0.4125400000256845) q[4];
ry(-0.003182064298058229) q[5];
rz(-0.6172705187757712) q[5];
ry(-0.00325552033332066) q[6];
rz(-0.5082461848989208) q[6];
ry(-0.01429929811570574) q[7];
rz(-1.4404147180865152) q[7];
ry(2.728296056039491) q[8];
rz(-3.0759393522944034) q[8];
ry(3.131102719701565) q[9];
rz(0.09159781717785975) q[9];
ry(-2.715809301154647) q[10];
rz(-0.9389648412197508) q[10];
ry(-3.1413991594238713) q[11];
rz(2.621025286656116) q[11];
ry(-0.9083963612679131) q[12];
rz(-2.278550391861751) q[12];
ry(-2.7141118731401686) q[13];
rz(0.1563269126048745) q[13];
ry(-1.3597021630500559) q[14];
rz(1.2757990735110345) q[14];
ry(-3.0863124158102853) q[15];
rz(-0.6366260782497051) q[15];
ry(0.0029594459109310023) q[16];
rz(-1.99325993197394) q[16];
ry(3.1382961364688073) q[17];
rz(2.1165643943670176) q[17];
ry(-1.4079157076292164) q[18];
rz(0.21103365557944304) q[18];
ry(-3.1326413295761006) q[19];
rz(0.4999565788688639) q[19];
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
ry(1.1027438426536555) q[0];
rz(0.7206224716138143) q[0];
ry(-0.016505526122989143) q[1];
rz(1.1665913573975493) q[1];
ry(-0.005046955790519725) q[2];
rz(1.5308874658502376) q[2];
ry(0.01771084897929353) q[3];
rz(0.11077450587415338) q[3];
ry(2.8505149509223653) q[4];
rz(0.2918481753539153) q[4];
ry(0.00025094929837976565) q[5];
rz(-1.097570833362556) q[5];
ry(1.7848324907121906) q[6];
rz(1.9240957130463776) q[6];
ry(-3.1321524868806256) q[7];
rz(1.7469721355738672) q[7];
ry(-2.7156606912939725) q[8];
rz(-2.9809198907859566) q[8];
ry(-1.5740349012609738) q[9];
rz(-2.020090255136666) q[9];
ry(2.5430353435212276) q[10];
rz(0.2614747675716126) q[10];
ry(0.00033375352733333585) q[11];
rz(-2.658058338959525) q[11];
ry(1.7768779315656815) q[12];
rz(2.960023437677152) q[12];
ry(3.0359624635807245) q[13];
rz(-1.9241832206210292) q[13];
ry(-3.1032026258416834) q[14];
rz(2.3743599152663992) q[14];
ry(-3.1066255966486667) q[15];
rz(-0.8418724198352358) q[15];
ry(3.114484371322394) q[16];
rz(-0.42562348448185716) q[16];
ry(-0.04273585455511241) q[17];
rz(-0.4588096756470805) q[17];
ry(-2.8409209469466012) q[18];
rz(0.19944476062269256) q[18];
ry(1.8957939508445791) q[19];
rz(2.9040165003511813) q[19];
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
ry(-2.219787052290191) q[0];
rz(0.6052680434688671) q[0];
ry(-1.7615506702433636) q[1];
rz(-0.5084217495178157) q[1];
ry(2.5793429680657214) q[2];
rz(-0.3643293467866435) q[2];
ry(1.522676281309498) q[3];
rz(1.09573804426245) q[3];
ry(-1.5174419466347275) q[4];
rz(-1.554911266084857) q[4];
ry(0.0016682808408689561) q[5];
rz(-0.5296557000394628) q[5];
ry(-3.1398504083216285) q[6];
rz(-1.4004715856541734) q[6];
ry(-3.098422942809636) q[7];
rz(-0.36324182252778137) q[7];
ry(0.007537522358673776) q[8];
rz(1.6197571848167094) q[8];
ry(3.0957531703442815) q[9];
rz(0.908226198015667) q[9];
ry(-0.33934390077977294) q[10];
rz(-3.081004197781728) q[10];
ry(0.001029979566652628) q[11];
rz(-0.6234079215534596) q[11];
ry(2.6162870948429973) q[12];
rz(1.5860561195893599) q[12];
ry(-1.8660780599439342) q[13];
rz(0.5253788813270815) q[13];
ry(-1.656519663431875) q[14];
rz(0.6232327480830812) q[14];
ry(2.916297803906918) q[15];
rz(-0.9293915654752466) q[15];
ry(0.003615106814128268) q[16];
rz(1.7141308554954255) q[16];
ry(-3.1380497050709883) q[17];
rz(-2.21400001496686) q[17];
ry(2.49026709271725) q[18];
rz(-1.8114360288661633) q[18];
ry(-1.851338022807916) q[19];
rz(-0.9766726892272324) q[19];
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
ry(-0.0515879122006603) q[0];
rz(0.6746873551439158) q[0];
ry(-0.8757002947083539) q[1];
rz(-2.7504266657285323) q[1];
ry(-3.093147506198006) q[2];
rz(-0.3043458269371815) q[2];
ry(1.5735993717619987) q[3];
rz(2.6975808849799248) q[3];
ry(-2.747145290519142) q[4];
rz(-1.641195292394821) q[4];
ry(-0.0015804182132832525) q[5];
rz(0.058955080853597686) q[5];
ry(-0.02826348432456324) q[6];
rz(-2.2115108519912026) q[6];
ry(-0.005461534077096246) q[7];
rz(-2.721567678944982) q[7];
ry(1.5863968187903472) q[8];
rz(-2.6684261437075705) q[8];
ry(0.0021225857739048948) q[9];
rz(-0.682927382051612) q[9];
ry(1.5111959146067313) q[10];
rz(-1.231036819115702) q[10];
ry(3.1414681308623913) q[11];
rz(-1.4423868268474982) q[11];
ry(1.564796209485142) q[12];
rz(1.4870348542542602) q[12];
ry(0.06622776901757772) q[13];
rz(0.9663722287770785) q[13];
ry(0.03508724748528547) q[14];
rz(-0.5902698208017974) q[14];
ry(-3.1370470107899693) q[15];
rz(-0.7821124676037607) q[15];
ry(-0.11566656030238409) q[16];
rz(1.2320613290448872) q[16];
ry(-3.126022967829047) q[17];
rz(-1.9388677287517264) q[17];
ry(1.7256919764031036) q[18];
rz(0.4760011647788023) q[18];
ry(1.001149962955927) q[19];
rz(-0.3152173111226402) q[19];
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
ry(3.056524277385977) q[0];
rz(1.4217316108555997) q[0];
ry(0.31269568063456804) q[1];
rz(2.3589230023872827) q[1];
ry(0.0014974162705136962) q[2];
rz(-0.7170653595880436) q[2];
ry(-3.1380884374223963) q[3];
rz(-2.40507249047013) q[3];
ry(0.06153449967080694) q[4];
rz(-1.49370171010286) q[4];
ry(1.5735129535485886) q[5];
rz(3.059054822137632) q[5];
ry(0.00025322602642763314) q[6];
rz(-2.3059056160220224) q[6];
ry(-3.0987130049117217) q[7];
rz(0.890583723059814) q[7];
ry(-1.4803462951811204) q[8];
rz(2.6424128950445414) q[8];
ry(-1.6488870446558668) q[9];
rz(2.2962891346289274) q[9];
ry(3.136645287883376) q[10];
rz(-1.6854302036851359) q[10];
ry(0.7806912549778914) q[11];
rz(1.908130004309885) q[11];
ry(1.4887378506024316) q[12];
rz(-0.18975266618379957) q[12];
ry(1.6137352577430857) q[13];
rz(1.292604175471749) q[13];
ry(-1.5130356649581487) q[14];
rz(-1.5848098853806885) q[14];
ry(0.1448693364297471) q[15];
rz(-1.4206124450582154) q[15];
ry(0.048918032664185755) q[16];
rz(3.087911884203523) q[16];
ry(0.0019191925813171896) q[17];
rz(1.2026069597480253) q[17];
ry(3.0812890155647925) q[18];
rz(2.4910909846226574) q[18];
ry(0.8606487927030724) q[19];
rz(-1.5965146620894552) q[19];
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
ry(0.22286058053370328) q[0];
rz(-1.2901015258134725) q[0];
ry(-1.7349077536647188) q[1];
rz(1.2910978692748891) q[1];
ry(1.547491962241909) q[2];
rz(1.6083575256010272) q[2];
ry(-0.0001305254641730258) q[3];
rz(1.6671550501583283) q[3];
ry(1.5683081629861548) q[4];
rz(1.978868025356861) q[4];
ry(-0.008828870070528792) q[5];
rz(0.16435032474331848) q[5];
ry(-1.609372632605753) q[6];
rz(-0.09324953659319295) q[6];
ry(3.1312286746513207) q[7];
rz(-1.2523714796785115) q[7];
ry(-0.002307922004373353) q[8];
rz(-2.321445303856596) q[8];
ry(-0.002614798033265775) q[9];
rz(-2.3144683835239386) q[9];
ry(-4.451631323619556e-05) q[10];
rz(-2.0864418628536363) q[10];
ry(0.0016785327971717836) q[11];
rz(-2.434878639047923) q[11];
ry(2.77714655350501) q[12];
rz(2.993741957721191) q[12];
ry(0.014710221879556125) q[13];
rz(1.19497353246143) q[13];
ry(-0.19569697697390873) q[14];
rz(-0.23340728197784788) q[14];
ry(-1.9870356746660054) q[15];
rz(-1.7641832104521686) q[15];
ry(-3.1414802490835183) q[16];
rz(-2.129530295915913) q[16];
ry(-0.07009711285451381) q[17];
rz(2.7314488696301047) q[17];
ry(-1.186296783586035) q[18];
rz(-2.1305552555218936) q[18];
ry(-1.8933488412575397) q[19];
rz(-3.1265283204446597) q[19];
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
ry(0.0002146553922273142) q[0];
rz(1.3865910097154535) q[0];
ry(2.4276436228539295) q[1];
rz(-0.8732780989297462) q[1];
ry(-2.340716868862791) q[2];
rz(-1.5739298911103226) q[2];
ry(-3.1407673046062086) q[3];
rz(-1.431832986265979) q[3];
ry(-1.0204667617585983e-06) q[4];
rz(1.6114142071448694) q[4];
ry(1.9841576263744027) q[5];
rz(2.219671367502057) q[5];
ry(-0.03582177219776041) q[6];
rz(0.19432661380331862) q[6];
ry(3.111030788594101) q[7];
rz(-2.9185325899135246) q[7];
ry(-1.6175370435752745) q[8];
rz(1.4019402970868526) q[8];
ry(-3.1350589971093394) q[9];
rz(-1.1066999860965634) q[9];
ry(0.0027492705235596664) q[10];
rz(-1.972166174125476) q[10];
ry(2.4492460122119155) q[11];
rz(-2.465403213110117) q[11];
ry(0.19652627509023238) q[12];
rz(-3.0449086248247093) q[12];
ry(-0.03179097475372797) q[13];
rz(-3.019213816342017) q[13];
ry(-0.020179852310780448) q[14];
rz(-2.245096104660407) q[14];
ry(0.014005811681346585) q[15];
rz(-1.3572069750429916) q[15];
ry(-0.009799543101387087) q[16];
rz(0.16234842758931156) q[16];
ry(0.005880380241762851) q[17];
rz(-1.1599209495683438) q[17];
ry(0.34530208491602793) q[18];
rz(-2.8029225395444133) q[18];
ry(-0.661770757326476) q[19];
rz(-2.517754108743599) q[19];
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
ry(-0.14419808064023254) q[0];
rz(0.6594795913331915) q[0];
ry(2.190461887920504) q[1];
rz(-2.3385811752300434) q[1];
ry(-1.571605214997379) q[2];
rz(-2.9771617416563703) q[2];
ry(-0.010970940725937517) q[3];
rz(2.9159471309151126) q[3];
ry(-3.1368621259022054) q[4];
rz(-0.4090522380932756) q[4];
ry(-0.01204360929112443) q[5];
rz(-0.9005736582992381) q[5];
ry(0.07868314044508111) q[6];
rz(0.021098625630271428) q[6];
ry(-3.1415270555936323) q[7];
rz(-1.1566585256241302) q[7];
ry(3.1405185861470732) q[8];
rz(2.0357170574301833) q[8];
ry(-0.001156149859170378) q[9];
rz(-2.323205581556793) q[9];
ry(-3.141324116055634) q[10];
rz(1.5911497125203635) q[10];
ry(-3.1409285093918755) q[11];
rz(3.117387979709038) q[11];
ry(-1.1354502496661543) q[12];
rz(-2.202397341426051) q[12];
ry(0.3060032568306799) q[13];
rz(-2.137768077848368) q[13];
ry(-1.060372179483947) q[14];
rz(-0.5221253809658597) q[14];
ry(-0.19995221941528651) q[15];
rz(2.3339784545415427) q[15];
ry(3.125405232396777) q[16];
rz(-2.190266320413876) q[16];
ry(-1.6071004993241464) q[17];
rz(-1.6690668504366868) q[17];
ry(0.29287330703259773) q[18];
rz(-0.31076049063450484) q[18];
ry(-2.9560654986548007) q[19];
rz(-0.7580854713372727) q[19];
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
ry(-1.5324641060428796) q[0];
rz(-1.7798672309555827) q[0];
ry(-1.312168723992536) q[1];
rz(-2.441041299627098) q[1];
ry(0.0035458747285202945) q[2];
rz(1.8404447625398186) q[2];
ry(-0.5458743018301216) q[3];
rz(-2.9878496317039587) q[3];
ry(3.1356287580233206) q[4];
rz(-1.6418451932908615) q[4];
ry(3.070433140191446) q[5];
rz(2.895580034477349) q[5];
ry(1.5335185512156881) q[6];
rz(3.12453266664579) q[6];
ry(-0.03454614212341252) q[7];
rz(3.1218869497849244) q[7];
ry(-1.8855380697111275) q[8];
rz(0.25965331868220676) q[8];
ry(1.5257092261951788) q[9];
rz(-0.8971594891049579) q[9];
ry(-0.0324832835757336) q[10];
rz(0.44888270021137083) q[10];
ry(-0.0720343112215028) q[11];
rz(2.6791569855597888) q[11];
ry(3.1076658409297186) q[12];
rz(2.3676852548041247) q[12];
ry(3.0776954670288084) q[13];
rz(1.0444991858403154) q[13];
ry(-3.1108971316643723) q[14];
rz(-1.6685075555347346) q[14];
ry(3.1398330307556304) q[15];
rz(0.225604577966693) q[15];
ry(3.141332959583368) q[16];
rz(0.5341751640048827) q[16];
ry(-2.3976559195426077) q[17];
rz(3.057418979642979) q[17];
ry(2.68204948871644) q[18];
rz(2.197446686363051) q[18];
ry(0.05884189087993409) q[19];
rz(-1.4046550345104845) q[19];
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
ry(-2.1139682655857506) q[0];
rz(-2.4789435037054997) q[0];
ry(-3.111647265910853) q[1];
rz(-1.8998018192873836) q[1];
ry(0.031039900257525233) q[2];
rz(1.7885409482856263) q[2];
ry(1.4316438879693587) q[3];
rz(2.8113889116660906) q[3];
ry(-1.5983137711454742) q[4];
rz(-0.6194670268041893) q[4];
ry(-1.6174295867224417) q[5];
rz(2.8521311518316144) q[5];
ry(3.102341499593971) q[6];
rz(-1.5800650968667371) q[6];
ry(0.0005912825933718224) q[7];
rz(-0.8527850986123475) q[7];
ry(3.1410498562492535) q[8];
rz(1.2673589580195657) q[8];
ry(-3.1377641987286315) q[9];
rz(-1.8006595931799572) q[9];
ry(-0.00045061395681469355) q[10];
rz(-1.6875612507967999) q[10];
ry(-3.1412260533581593) q[11];
rz(-2.527569385529438) q[11];
ry(0.15061843437820463) q[12];
rz(-1.4527164871766098) q[12];
ry(-2.8324888319282415) q[13];
rz(0.5167237607718391) q[13];
ry(-2.2554642161694174) q[14];
rz(-2.372677829555257) q[14];
ry(2.852633464577423) q[15];
rz(-1.9967657315844203) q[15];
ry(-3.1315840410305458) q[16];
rz(-2.353264918603458) q[16];
ry(2.232709680535092) q[17];
rz(3.097354260840174) q[17];
ry(-2.559573751085671) q[18];
rz(2.5105930459883874) q[18];
ry(3.132425745719117) q[19];
rz(-2.0653946237175376) q[19];
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
ry(-0.07162434558566348) q[0];
rz(2.4275987559935785) q[0];
ry(-0.011134964959648386) q[1];
rz(0.9565837648890643) q[1];
ry(-0.000521132334708894) q[2];
rz(-2.8243554285352594) q[2];
ry(-0.005739944369582695) q[3];
rz(0.9758029569885459) q[3];
ry(3.141071242380645) q[4];
rz(-2.214306609705494) q[4];
ry(3.1414502833986435) q[5];
rz(2.8535761993629802) q[5];
ry(-1.5666630382525044) q[6];
rz(-1.5748155649198303) q[6];
ry(-2.5337065498487905) q[7];
rz(1.739168765177686) q[7];
ry(1.1053597349576063) q[8];
rz(0.23243515241702165) q[8];
ry(-1.381367201843184) q[9];
rz(-0.8958126237691004) q[9];
ry(0.17395651399963225) q[10];
rz(3.0017218831570998) q[10];
ry(-0.3263695251199575) q[11];
rz(2.0052661342551805) q[11];
ry(-1.7030990216721076) q[12];
rz(-0.04106316475364856) q[12];
ry(-0.43146947370189886) q[13];
rz(-1.5107762298529153) q[13];
ry(-0.018719238780080645) q[14];
rz(1.1896817260228485) q[14];
ry(0.0008179544739349689) q[15];
rz(-1.661310693624837) q[15];
ry(-3.1415715519636773) q[16];
rz(1.0202908465797547) q[16];
ry(2.217895833502218) q[17];
rz(0.4725540137018972) q[17];
ry(-1.7954335334394633) q[18];
rz(-1.801961537089353) q[18];
ry(0.9429605867437408) q[19];
rz(1.2632048278688135) q[19];
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
ry(1.6027693923112736) q[0];
rz(0.6059371639420658) q[0];
ry(3.0139225159192047) q[1];
rz(-2.905119731564425) q[1];
ry(0.8970691715511042) q[2];
rz(-0.03056174743672821) q[2];
ry(0.17206439119294017) q[3];
rz(-2.064801711246154) q[3];
ry(-1.3798667298918663) q[4];
rz(-2.3472803717978157) q[4];
ry(-1.50511054497219) q[5];
rz(-1.5039271327938295) q[5];
ry(-1.5668147924627986) q[6];
rz(2.8949648208708463) q[6];
ry(1.5366017494670412) q[7];
rz(-1.3809425866198772) q[7];
ry(1.1329221722793603) q[8];
rz(-0.9050260942610214) q[8];
ry(-3.141419631867483) q[9];
rz(2.350156009823342) q[9];
ry(3.1411915810923166) q[10];
rz(2.3379933151867514) q[10];
ry(-1.7379372724105098) q[11];
rz(-1.4464647101845203) q[11];
ry(-1.6501642692483691) q[12];
rz(-3.0692189741800964) q[12];
ry(-1.5660848194514536) q[13];
rz(-0.002149434428129205) q[13];
ry(-1.2107176729227591) q[14];
rz(2.4761766617978735) q[14];
ry(-1.3792780449535966) q[15];
rz(-2.9042466008348313) q[15];
ry(3.0979218543180527) q[16];
rz(-0.5278508485544554) q[16];
ry(1.3470020108825478) q[17];
rz(0.8388973773674165) q[17];
ry(1.8400377036602173) q[18];
rz(-0.7103805695559097) q[18];
ry(2.0864386350012856) q[19];
rz(-1.4959751334958034) q[19];
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
ry(3.1369083360664147) q[0];
rz(-2.813833313996055) q[0];
ry(1.5806375400186636) q[1];
rz(2.4708517861437467) q[1];
ry(-1.23002069372403) q[2];
rz(-1.5625201440256076) q[2];
ry(-0.0189893869156926) q[3];
rz(-0.4056362117806174) q[3];
ry(0.0003484256577753153) q[4];
rz(2.4710284102764826) q[4];
ry(3.1409880660826066) q[5];
rz(1.7005460561439485) q[5];
ry(-3.141157532437639) q[6];
rz(3.052865763623979) q[6];
ry(-1.8427833390336168) q[7];
rz(-0.006673003389974675) q[7];
ry(-3.1358616886537023) q[8];
rz(1.6589510821829145) q[8];
ry(-5.853344139204709e-05) q[9];
rz(-2.2056622055628496) q[9];
ry(0.009736440437048218) q[10];
rz(-2.00399000631649) q[10];
ry(2.78485236803896) q[11];
rz(1.686782584071607) q[11];
ry(1.4117660424465948) q[12];
rz(-2.4310572458566555) q[12];
ry(-1.5805150845444045) q[13];
rz(-3.0265732169125545) q[13];
ry(0.008525017119915823) q[14];
rz(-1.0009956039534793) q[14];
ry(3.1410827720987387) q[15];
rz(1.9213647540387786) q[15];
ry(3.1393061928134767) q[16];
rz(-2.3353041605394558) q[16];
ry(0.864976712236724) q[17];
rz(-0.025444256730239048) q[17];
ry(0.30265331223229275) q[18];
rz(0.3320038956611526) q[18];
ry(-1.2361972956365797) q[19];
rz(-1.1247189656644982) q[19];
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
ry(-3.118047970060069) q[0];
rz(0.10360461452521211) q[0];
ry(3.1389776582610067) q[1];
rz(2.4323675163694403) q[1];
ry(1.5796171482373227) q[2];
rz(-2.434618674163261) q[2];
ry(-0.02703956154257328) q[3];
rz(-3.068245860991699) q[3];
ry(2.558109426644233) q[4];
rz(-1.5711692104077264) q[4];
ry(-3.119627501998862) q[5];
rz(2.086216161907481) q[5];
ry(3.126416060028045) q[6];
rz(0.9353586047421153) q[6];
ry(-2.9709979648329994) q[7];
rz(2.50599164361269) q[7];
ry(0.5094891919897901) q[8];
rz(-2.6874865209129952) q[8];
ry(0.0016417666362493434) q[9];
rz(-2.297477399732089) q[9];
ry(0.0034422167903459767) q[10];
rz(1.1729609280991884) q[10];
ry(-1.627634400504326) q[11];
rz(2.6236816589834144) q[11];
ry(-0.0006641658084782111) q[12];
rz(1.7594463115409882) q[12];
ry(-0.08097868626890214) q[13];
rz(-0.8672715732170647) q[13];
ry(-1.4709413772945412) q[14];
rz(2.759385031111116) q[14];
ry(-0.0012558070497853933) q[15];
rz(0.2429989730929947) q[15];
ry(3.136656172662912) q[16];
rz(-1.7440772494151935) q[16];
ry(-1.9580072338608208) q[17];
rz(3.1407243202340775) q[17];
ry(-0.13343313386274858) q[18];
rz(-0.7590251824969041) q[18];
ry(-3.057038974325015) q[19];
rz(0.2760826896116682) q[19];
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
ry(3.0811066291390516) q[0];
rz(0.0962774285697169) q[0];
ry(1.571488163387074) q[1];
rz(-0.17709169732061716) q[1];
ry(1.5734122995241977) q[2];
rz(-1.6011805782330166) q[2];
ry(-0.00047558955954099167) q[3];
rz(-1.08266415185639) q[3];
ry(-2.5807002750195185) q[4];
rz(-0.17731779396019487) q[4];
ry(0.0008305047658702617) q[5];
rz(-1.0923693090846855) q[5];
ry(-3.141455537386661) q[6];
rz(-2.405253734548244) q[6];
ry(-0.3315335379185766) q[7];
rz(-2.4857189372036927) q[7];
ry(3.114086623285362) q[8];
rz(-0.0403289948000911) q[8];
ry(1.5727748275282385) q[9];
rz(-1.3369227006994828) q[9];
ry(-3.1412396359929757) q[10];
rz(2.150798002762995) q[10];
ry(-3.0913039668818647) q[11];
rz(-0.6293114384063145) q[11];
ry(0.5190063137868055) q[12];
rz(1.6487814610661131) q[12];
ry(-0.006823247726403723) q[13];
rz(-2.38795871145638) q[13];
ry(-0.042685814897210725) q[14];
rz(1.9736364441373186) q[14];
ry(0.0012398461075999734) q[15];
rz(2.8349560327317773) q[15];
ry(3.1415490481632062) q[16];
rz(2.3558004357107434) q[16];
ry(-2.2883355701101347) q[17];
rz(1.2243192386366875) q[17];
ry(-1.8655487526870331) q[18];
rz(-0.03400563493698308) q[18];
ry(-1.7958811239808705) q[19];
rz(2.9199144438928792) q[19];
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
ry(0.2163703934874981) q[0];
rz(1.7131503804045822) q[0];
ry(-0.023693217768382624) q[1];
rz(-1.5090461916175952) q[1];
ry(1.2265440099181761) q[2];
rz(-0.0970964976959019) q[2];
ry(-3.0190891886011944) q[3];
rz(-0.9436067903311329) q[3];
ry(3.122274783520361) q[4];
rz(-1.10252284741762) q[4];
ry(-0.006828817658733762) q[5];
rz(1.916245145928431) q[5];
ry(-3.0149584748489113) q[6];
rz(0.0004178823444130713) q[6];
ry(-0.8572882169316052) q[7];
rz(-1.4167292623001035) q[7];
ry(1.5439153262839547) q[8];
rz(-1.3703978156777044) q[8];
ry(3.132463308158163) q[9];
rz(1.9033208397694117) q[9];
ry(0.000385613389867423) q[10];
rz(1.993452457072431) q[10];
ry(1.5705173517143083) q[11];
rz(-2.8383525970390426) q[11];
ry(-0.00014070529518139807) q[12];
rz(-2.696960756820131) q[12];
ry(-1.5125508142868194) q[13];
rz(1.567203001073093) q[13];
ry(1.635262710000239) q[14];
rz(1.4183830913656887) q[14];
ry(-0.08983287201472034) q[15];
rz(-1.4747708904943804) q[15];
ry(-0.0001185283880511534) q[16];
rz(-0.5196887246007167) q[16];
ry(-0.49813929247520383) q[17];
rz(-1.0501375138404905) q[17];
ry(1.0051200323361753) q[18];
rz(-2.6798668230653444) q[18];
ry(1.6726494013452398) q[19];
rz(0.8311370098094786) q[19];
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
ry(2.1867866746352567) q[0];
rz(0.19421638470119165) q[0];
ry(-1.242481741814291) q[1];
rz(-2.0125341849576133) q[1];
ry(-3.101528216166105) q[2];
rz(0.053468108316277345) q[2];
ry(-3.1410891052779144) q[3];
rz(1.9244303809087562) q[3];
ry(3.1064074046957826) q[4];
rz(-1.072067094856175) q[4];
ry(3.1415767860530064) q[5];
rz(-1.2892665039622206) q[5];
ry(-3.1278966590789445) q[6];
rz(1.3477498568071207) q[6];
ry(1.659963218525462e-05) q[7];
rz(2.9481765960178126) q[7];
ry(0.1502755047265838) q[8];
rz(2.1836691423032035) q[8];
ry(2.289721585668758e-07) q[9];
rz(-0.09881412347054469) q[9];
ry(0.011546368799227658) q[10];
rz(0.39507827641483956) q[10];
ry(-0.0010367289715667155) q[11];
rz(2.8584769380604453) q[11];
ry(1.5774177584442386) q[12];
rz(2.2438667868222444) q[12];
ry(-1.5696522096590746) q[13];
rz(2.5065646246844824) q[13];
ry(1.565471495166524) q[14];
rz(-1.690171204912894) q[14];
ry(-3.141575537468272) q[15];
rz(-1.4248229061473134) q[15];
ry(0.0005809623223133996) q[16];
rz(-0.4873640516892151) q[16];
ry(-3.1059280636771587) q[17];
rz(0.8561494004012875) q[17];
ry(0.227847095306557) q[18];
rz(-0.22165444417984312) q[18];
ry(2.759635383852455) q[19];
rz(-1.218992110187334) q[19];
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
ry(-0.0005697390855790019) q[0];
rz(-3.101993747995686) q[0];
ry(-3.140805616459277) q[1];
rz(-0.4378170498129722) q[1];
ry(-0.008382664297672242) q[2];
rz(1.4212252086479706) q[2];
ry(-2.779743668163406) q[3];
rz(-0.6965384083576414) q[3];
ry(-0.019523283930173108) q[4];
rz(-2.8776143815513606) q[4];
ry(3.1346237678517963) q[5];
rz(-0.9275715524642147) q[5];
ry(-0.005545687994450664) q[6];
rz(-1.3192061547651432) q[6];
ry(-1.3732515114185118) q[7];
rz(1.885764834682222) q[7];
ry(-3.1390173216521915) q[8];
rz(-1.4028782581110069) q[8];
ry(-1.574878427850678) q[9];
rz(-0.6506683831739176) q[9];
ry(1.5717966138325257) q[10];
rz(0.005789839069366033) q[10];
ry(0.2043855631626787) q[11];
rz(3.131503814035258) q[11];
ry(0.00034036737726630625) q[12];
rz(2.558030216938684) q[12];
ry(3.13884488415558) q[13];
rz(-2.7874081208919383) q[13];
ry(3.137793488374157) q[14];
rz(-0.8117356888400903) q[14];
ry(-1.546390727388201) q[15];
rz(2.915460419585847) q[15];
ry(1.570378630294077) q[16];
rz(1.5366639974709422) q[16];
ry(2.741470437625702) q[17];
rz(2.773588833568833) q[17];
ry(1.4872748422018938) q[18];
rz(2.9056160753956535) q[18];
ry(2.9457098787871323) q[19];
rz(-2.0772726811918254) q[19];
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
ry(2.510685973863294) q[0];
rz(-2.5721882687957653) q[0];
ry(-1.5744155006684606) q[1];
rz(2.0883379383572276) q[1];
ry(-1.5675577761627908) q[2];
rz(1.4150787679038688) q[2];
ry(0.0013869935669982283) q[3];
rz(2.3556034003051862) q[3];
ry(0.2317832726626996) q[4];
rz(3.0210465292421484) q[4];
ry(-3.141437644713019) q[5];
rz(-0.7448832657345084) q[5];
ry(0.08171548286160579) q[6];
rz(-1.5574156656568854) q[6];
ry(-0.0004402846908771529) q[7];
rz(-1.8574581707815048) q[7];
ry(-0.00373959055632988) q[8];
rz(2.2183732261491342) q[8];
ry(-0.00020030512517484912) q[9];
rz(0.6418752090920575) q[9];
ry(-3.128482311580074) q[10];
rz(1.5374517561351784) q[10];
ry(-1.341324068211816) q[11];
rz(3.1405121920988543) q[11];
ry(-1.5715033964668925) q[12];
rz(-1.7954039795246846) q[12];
ry(0.0006788050323159725) q[13];
rz(-2.6831344413277916) q[13];
ry(0.0006622699953049795) q[14];
rz(0.7070948457419471) q[14];
ry(0.0005043699271845314) q[15];
rz(-0.868065984146468) q[15];
ry(0.0011745072518333766) q[16];
rz(-1.54586840871514) q[16];
ry(3.070519731842481) q[17];
rz(-1.47356452201539) q[17];
ry(1.5687537787147976) q[18];
rz(-0.000617759118278431) q[18];
ry(-1.3244987531637218) q[19];
rz(2.88167193082064) q[19];
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
ry(-0.001560812385551734) q[0];
rz(-1.852537947371534) q[0];
ry(-3.1401383130846545) q[1];
rz(-0.635843694233655) q[1];
ry(0.00014987359212865288) q[2];
rz(1.8258169093264573) q[2];
ry(3.132379025033831) q[3];
rz(-2.9816919621257587) q[3];
ry(-1.5704830586752527) q[4];
rz(-1.445486283938575) q[4];
ry(-3.14000681875249) q[5];
rz(-2.336372945328015) q[5];
ry(-1.5719283678670442) q[6];
rz(0.244001478853505) q[6];
ry(0.15791669985577095) q[7];
rz(-0.7996595815246721) q[7];
ry(-1.5418129309305764) q[8];
rz(-3.019032515694917) q[8];
ry(1.5706664419936052) q[9];
rz(1.647118315969094) q[9];
ry(-0.00027529282840710323) q[10];
rz(-1.4290787854212763) q[10];
ry(0.24204337621035418) q[11];
rz(-1.4983869348888865) q[11];
ry(0.00016270285705477508) q[12];
rz(0.3296081750200721) q[12];
ry(0.04178553968324366) q[13];
rz(1.7681760043524675) q[13];
ry(-1.5706753245332532) q[14];
rz(0.09829915864197591) q[14];
ry(-0.05561871606096869) q[15];
rz(2.749319983566086) q[15];
ry(-3.12798520116404) q[16];
rz(-1.333716971715844) q[16];
ry(-3.1409168185935794) q[17];
rz(0.3606354569216954) q[17];
ry(1.2296339507481715) q[18];
rz(-1.322048774777431) q[18];
ry(2.3466759636317884) q[19];
rz(1.6592997248394248) q[19];