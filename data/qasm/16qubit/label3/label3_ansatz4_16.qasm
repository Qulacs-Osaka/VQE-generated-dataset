OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.6517994213294427) q[0];
rz(-3.0932164557150186) q[0];
ry(-1.5242441674362328) q[1];
rz(-0.0033291870222676117) q[1];
ry(0.21145615475740434) q[2];
rz(-2.517515922466659) q[2];
ry(-0.35736066540779143) q[3];
rz(-0.43144919247551317) q[3];
ry(-2.015156775002753) q[4];
rz(0.31588712267760016) q[4];
ry(2.0740217394687086) q[5];
rz(-1.7356244275399035) q[5];
ry(-3.127153528679864) q[6];
rz(-1.5746882166592586) q[6];
ry(-3.1315814603068555) q[7];
rz(-0.2730365586204781) q[7];
ry(1.5675049060406543) q[8];
rz(-3.118954529122815) q[8];
ry(-1.5658242113670653) q[9];
rz(1.8675450521334962) q[9];
ry(0.33013324274764) q[10];
rz(0.07772121187043268) q[10];
ry(2.6555693446121276) q[11];
rz(-1.2067510150308232) q[11];
ry(-1.676968755665575) q[12];
rz(1.5443583114027388) q[12];
ry(1.567949540090257) q[13];
rz(-3.1344893767243085) q[13];
ry(-3.140887463496468) q[14];
rz(-1.6262446135241067) q[14];
ry(-3.140682916375853) q[15];
rz(1.052847630567701) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.3827754487578341) q[0];
rz(-2.085187820549665) q[0];
ry(-1.9738395899625676) q[1];
rz(1.1068778942646116) q[1];
ry(-1.273605766067412) q[2];
rz(-0.7142243875092981) q[2];
ry(0.4214765316640706) q[3];
rz(0.7019736660198607) q[3];
ry(2.1044003520877874) q[4];
rz(1.6617333196584436) q[4];
ry(-2.345560365379669) q[5];
rz(2.4026233676376783) q[5];
ry(-0.8778960144176686) q[6];
rz(-0.10472834377717266) q[6];
ry(1.3534071585310752) q[7];
rz(3.042151260539226) q[7];
ry(-2.6760146331063974) q[8];
rz(-0.11370414951242709) q[8];
ry(-2.0652198478042147) q[9];
rz(-0.5301794655074482) q[9];
ry(2.8844131179423265) q[10];
rz(0.40874445323290026) q[10];
ry(2.636048515496466) q[11];
rz(-0.2985168653122496) q[11];
ry(3.136037946650404) q[12];
rz(-1.513477434576945) q[12];
ry(1.5681557730282512) q[13];
rz(3.081921817730006) q[13];
ry(-1.7800729623870053) q[14];
rz(-1.2024083311380624) q[14];
ry(-1.4362337623338686) q[15];
rz(-2.039160967554825) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.6791749754950847) q[0];
rz(-0.40021923838912343) q[0];
ry(1.5331046138604132) q[1];
rz(2.7251602728259963) q[1];
ry(-0.8476988137367708) q[2];
rz(-0.9467370632110412) q[2];
ry(-1.0305318020547372) q[3];
rz(-1.922653992642322) q[3];
ry(-2.2137093936349546) q[4];
rz(-0.8709117900081095) q[4];
ry(3.0192383901290922) q[5];
rz(-1.083805207279745) q[5];
ry(3.1028042511215355) q[6];
rz(-1.7693561403799674) q[6];
ry(0.04229168803939931) q[7];
rz(0.8944375728030313) q[7];
ry(3.139650157098241) q[8];
rz(1.4363111261567487) q[8];
ry(-3.1113558626856284) q[9];
rz(2.0449448333220674) q[9];
ry(-3.1369203186915677) q[10];
rz(0.1622235721895839) q[10];
ry(-3.1326087223689267) q[11];
rz(0.36886254489269654) q[11];
ry(1.5750717218098904) q[12];
rz(-3.138196055425038) q[12];
ry(-1.568451908590692) q[13];
rz(-3.140075894199627) q[13];
ry(2.164616879417478) q[14];
rz(-1.3822809739779116) q[14];
ry(-3.1074859846809586) q[15];
rz(2.258324263883906) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.30576080593438587) q[0];
rz(-0.31993952843514895) q[0];
ry(0.29463602712803133) q[1];
rz(-1.1779888682180464) q[1];
ry(0.8115171091587755) q[2];
rz(0.16651731608860487) q[2];
ry(-0.9013093181070078) q[3];
rz(-0.441918109826533) q[3];
ry(1.6328144724346094) q[4];
rz(2.616511454507526) q[4];
ry(1.3312437245453845) q[5];
rz(-1.755587025674128) q[5];
ry(-2.909012043794358) q[6];
rz(1.931005188053926) q[6];
ry(-0.6123945338173783) q[7];
rz(0.023888768977924357) q[7];
ry(-1.5788019233334296) q[8];
rz(0.7344149334081537) q[8];
ry(-0.572435084783514) q[9];
rz(-2.4554622122421876) q[9];
ry(-2.458927075354832) q[10];
rz(-2.1195804012318726) q[10];
ry(-2.806958072589667) q[11];
rz(1.6572996241234534) q[11];
ry(-0.8230174603433474) q[12];
rz(-1.5687693065994572) q[12];
ry(2.1933910975229236) q[13];
rz(1.5790128465761621) q[13];
ry(-0.8458115824047274) q[14];
rz(-1.4070631042852073) q[14];
ry(-0.5350857833057683) q[15];
rz(1.4030457207330982) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.29189973863963886) q[0];
rz(1.0920506471662286) q[0];
ry(-2.97462920924556) q[1];
rz(1.2835177844077887) q[1];
ry(1.3446954048588224) q[2];
rz(1.2112452492456738) q[2];
ry(1.5525352628914095) q[3];
rz(-0.5412857251965334) q[3];
ry(-1.8708258860188824) q[4];
rz(1.1176637472958109) q[4];
ry(1.418443423234453) q[5];
rz(1.2261927954361729) q[5];
ry(-0.6682848389600569) q[6];
rz(1.4407664763607642) q[6];
ry(-2.925693579124359) q[7];
rz(0.3702959615094308) q[7];
ry(-2.9100131654235693e-05) q[8];
rz(1.47786580417218) q[8];
ry(-3.1412483134607223) q[9];
rz(-1.499315684170532) q[9];
ry(-0.37833941900368384) q[10];
rz(1.0937569491170884) q[10];
ry(-1.8697426463774969) q[11];
rz(0.5213331771619089) q[11];
ry(-0.3171236849251058) q[12];
rz(-1.5552023535828159) q[12];
ry(-2.394903598199277) q[13];
rz(-1.407311262364558) q[13];
ry(3.0687166828157615) q[14];
rz(-1.9314783439362448) q[14];
ry(1.039883557285209) q[15];
rz(1.60741198450457) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.9299606467517862) q[0];
rz(0.2535217159811914) q[0];
ry(2.931918235044075) q[1];
rz(2.9630782759292615) q[1];
ry(2.2650947984321226) q[2];
rz(0.6293208701769588) q[2];
ry(2.019821801752025) q[3];
rz(1.9367371320205813) q[3];
ry(0.163086615693703) q[4];
rz(-0.8208754588289805) q[4];
ry(-0.9468452473270376) q[5];
rz(-1.5260746079079452) q[5];
ry(1.2563774523797067) q[6];
rz(-0.024413801640377916) q[6];
ry(-1.8727153260601952) q[7];
rz(-1.5382509324504337) q[7];
ry(-0.004055896211679654) q[8];
rz(1.9925799330540928) q[8];
ry(0.004028763777090383) q[9];
rz(-2.430502996971573) q[9];
ry(1.7316996270476714) q[10];
rz(-1.370883445493611) q[10];
ry(-0.9076109317172651) q[11];
rz(-0.617617504905666) q[11];
ry(-3.0047422392709136) q[12];
rz(1.5979208948609267) q[12];
ry(3.1024401013730762) q[13];
rz(-1.4057483992026114) q[13];
ry(-2.6658010360500444) q[14];
rz(1.1678447369228697) q[14];
ry(1.8827369140392882) q[15];
rz(-1.854376222634716) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.4482584795755704) q[0];
rz(-1.1981049929526186) q[0];
ry(1.3626008090635364) q[1];
rz(-1.2755024018282632) q[1];
ry(1.403184120568504) q[2];
rz(2.6000928329947426) q[2];
ry(-1.4169570606179318) q[3];
rz(-2.321537476981988) q[3];
ry(3.048940186388017) q[4];
rz(-1.2356651898000588) q[4];
ry(0.9350928518904427) q[5];
rz(1.8579717190345226) q[5];
ry(-0.3533913726435324) q[6];
rz(-0.29253480581154623) q[6];
ry(2.4885129360050247) q[7];
rz(2.2534979033752274) q[7];
ry(-0.0005156341922392116) q[8];
rz(3.099980189067755) q[8];
ry(-3.1402730496649798) q[9];
rz(-0.3973799315100639) q[9];
ry(-2.6647068577160655) q[10];
rz(-2.6050052544386766) q[10];
ry(-0.2834316357943436) q[11];
rz(2.9019681979302034) q[11];
ry(-0.9852604797213695) q[12];
rz(1.571977880062958) q[12];
ry(2.2847510023636275) q[13];
rz(-1.561385713169871) q[13];
ry(0.5886618197408637) q[14];
rz(1.834700706344793) q[14];
ry(-1.8828225499722595) q[15];
rz(-2.0731356397547094) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.5669545677164942) q[0];
rz(-0.007603960571780809) q[0];
ry(1.5820535384515644) q[1];
rz(-0.03015563733502002) q[1];
ry(1.4160938526530849) q[2];
rz(-0.7451713585618345) q[2];
ry(-1.9072501304098903) q[3];
rz(-0.4284137502746715) q[3];
ry(-2.1246788237954237) q[4];
rz(2.359418476098168) q[4];
ry(-1.1717759164589365) q[5];
rz(-1.6040930292069584) q[5];
ry(-2.9194759276453897) q[6];
rz(0.6073015578878421) q[6];
ry(1.1785615719385065) q[7];
rz(3.0353860450867165) q[7];
ry(3.1379405711593407) q[8];
rz(2.584060338347617) q[8];
ry(3.132295350374663) q[9];
rz(-2.0879632133447155) q[9];
ry(3.0560001714673506) q[10];
rz(-1.0983512020479658) q[10];
ry(2.518793503802789) q[11];
rz(-0.8592081706094534) q[11];
ry(-0.22678505646318112) q[12];
rz(-1.5296724911390913) q[12];
ry(-2.9268990503538896) q[13];
rz(-0.07453500734041096) q[13];
ry(1.1849418179500653) q[14];
rz(2.6968313512287407) q[14];
ry(0.5067370624093838) q[15];
rz(-0.7130729822018317) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.37330319045298843) q[0];
rz(1.9368165863003197) q[0];
ry(0.36652091904136963) q[1];
rz(1.4351479775680571) q[1];
ry(0.7692373197751142) q[2];
rz(1.275431542642315) q[2];
ry(-2.7571919769686732) q[3];
rz(-3.125917164079772) q[3];
ry(0.8944370450803931) q[4];
rz(1.0116871166841674) q[4];
ry(-2.448858884902547) q[5];
rz(2.8341700206906726) q[5];
ry(-2.1110819104983696) q[6];
rz(-2.826608522982967) q[6];
ry(-2.3449316181746758) q[7];
rz(2.1904663451323567) q[7];
ry(0.08426028718713546) q[8];
rz(-0.06933361403123951) q[8];
ry(-3.1096051971355005) q[9];
rz(1.9521467026293458) q[9];
ry(1.2871288781328722) q[10];
rz(-2.1496877054676182) q[10];
ry(-1.2307083988322898) q[11];
rz(1.836929962213489) q[11];
ry(-3.086199680962183) q[12];
rz(1.6127594549464364) q[12];
ry(3.1411424524920966) q[13];
rz(3.058660719640501) q[13];
ry(-1.6834604915233362) q[14];
rz(-2.737918158085376) q[14];
ry(1.743468926343226) q[15];
rz(-0.7414917945483772) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.737689695408481) q[0];
rz(0.27477660544226973) q[0];
ry(-0.409686944284242) q[1];
rz(-3.1008731306491044) q[1];
ry(-2.1391912040502055) q[2];
rz(-1.7765311420498726) q[2];
ry(0.5912291443881964) q[3];
rz(-0.9268742428741251) q[3];
ry(-1.9935381649178159) q[4];
rz(2.468689470121399) q[4];
ry(2.6598959844206287) q[5];
rz(-2.1288577410628227) q[5];
ry(-1.4556634339172607) q[6];
rz(1.1403839955149457) q[6];
ry(-1.8624255900863385) q[7];
rz(-1.345365170394384) q[7];
ry(0.0010980769170028967) q[8];
rz(-2.7917806495925204) q[8];
ry(-3.141178034205385) q[9];
rz(-2.7776704909840744) q[9];
ry(-1.4623285040113805) q[10];
rz(-1.8972163564894462) q[10];
ry(-1.283866217743098) q[11];
rz(-1.412572378211574) q[11];
ry(-2.1831636531941467) q[12];
rz(1.5694528238616936) q[12];
ry(-0.6294114994721144) q[13];
rz(-1.5717251858776342) q[13];
ry(2.219696488891282) q[14];
rz(-2.227530208832791) q[14];
ry(-1.4780501652569216) q[15];
rz(-1.5610449886689306) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.620965878606869) q[0];
rz(-0.8775926983302397) q[0];
ry(-1.747194387967302) q[1];
rz(-1.1348441714789315) q[1];
ry(0.7728855312170604) q[2];
rz(-0.7854256864658914) q[2];
ry(2.1036599210681253) q[3];
rz(-0.0029235883687060795) q[3];
ry(-0.4027787819769397) q[4];
rz(2.0744332118054385) q[4];
ry(-0.20424131956372626) q[5];
rz(-1.3915805413625817) q[5];
ry(2.174901950041928) q[6];
rz(2.5675986638731625) q[6];
ry(1.7281907039218982) q[7];
rz(-0.02514623449389818) q[7];
ry(-3.141391002510755) q[8];
rz(-0.18814365010764345) q[8];
ry(1.5666088754690415) q[9];
rz(2.7058894450374957) q[9];
ry(-1.226790063326093) q[10];
rz(0.46413384718254963) q[10];
ry(1.8511015962934174) q[11];
rz(3.0190331149994334) q[11];
ry(0.15727569798382782) q[12];
rz(-1.4797228627803467) q[12];
ry(0.07976666798169418) q[13];
rz(-0.04279072793307659) q[13];
ry(0.7444999574961203) q[14];
rz(-2.867823107577838) q[14];
ry(3.0944315648450846) q[15];
rz(3.102166709301226) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.530191430742498) q[0];
rz(-2.667530658912006) q[0];
ry(0.39353836593818414) q[1];
rz(0.45313063926659863) q[1];
ry(-0.4829910186664765) q[2];
rz(-1.8990359681355802) q[2];
ry(1.9795137422431943) q[3];
rz(1.507223039994832) q[3];
ry(-2.306082518236524) q[4];
rz(2.56695777394794) q[4];
ry(1.898847435734229) q[5];
rz(0.9801196363816223) q[5];
ry(3.138067981197397) q[6];
rz(-1.6189379378808373) q[6];
ry(2.977204107396216) q[7];
rz(-2.8394814598553366) q[7];
ry(0.00477414554589739) q[8];
rz(-1.1019483450490009) q[8];
ry(3.13772501865857) q[9];
rz(2.7200951569444434) q[9];
ry(1.5672448475715655) q[10];
rz(1.561301933813673) q[10];
ry(0.01181606108910671) q[11];
rz(1.9055667984038314) q[11];
ry(-0.023272249797023845) q[12];
rz(-2.7087998660736177) q[12];
ry(-3.1379941162091805) q[13];
rz(-0.037187488509853324) q[13];
ry(-0.7018242427024749) q[14];
rz(-2.4233967000441288) q[14];
ry(2.637076929694328) q[15];
rz(-1.356388257474901) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.46860321575926184) q[0];
rz(-2.171430910169665) q[0];
ry(1.0541303617658362) q[1];
rz(0.061629441563446345) q[1];
ry(-2.9998300665112336) q[2];
rz(-2.4067977102722384) q[2];
ry(0.0487177359859734) q[3];
rz(0.9129939014348261) q[3];
ry(-3.019760149811361) q[4];
rz(-1.6710314299389122) q[4];
ry(2.918104857352715) q[5];
rz(0.6857410516480991) q[5];
ry(-2.8414148368302334) q[6];
rz(1.720388936457171) q[6];
ry(2.7736825452710736) q[7];
rz(1.2351315047740359) q[7];
ry(1.5128125143601905) q[8];
rz(2.283056918811617) q[8];
ry(-0.02664590138059797) q[9];
rz(0.9202132564834745) q[9];
ry(1.5813112549517312) q[10];
rz(2.374443870250554) q[10];
ry(1.5767974685336945) q[11];
rz(-3.128739011016026) q[11];
ry(1.5693031757855307) q[12];
rz(1.5854237307674346) q[12];
ry(1.5623410003425313) q[13];
rz(0.5362507814690982) q[13];
ry(1.7097041445068664) q[14];
rz(1.572582290473325) q[14];
ry(-0.7342680445467157) q[15];
rz(-3.0145367055028136) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.6137583381709397) q[0];
rz(-2.079134728042765) q[0];
ry(-1.174059243181493) q[1];
rz(0.6266205830471094) q[1];
ry(-2.774975490166008) q[2];
rz(0.4161785967229445) q[2];
ry(-0.3618661399741407) q[3];
rz(-2.78734449133552) q[3];
ry(2.245784647320958) q[4];
rz(1.5613372010263615) q[4];
ry(0.9289337639748121) q[5];
rz(1.9307806411905901) q[5];
ry(1.5711827203442252) q[6];
rz(3.1412430651583843) q[6];
ry(-1.5687897483330435) q[7];
rz(-0.004084492797761463) q[7];
ry(0.0019460699476212537) q[8];
rz(3.033716095103985) q[8];
ry(3.1388796528393414) q[9];
rz(-0.750988010580473) q[9];
ry(-1.6975787965556325e-05) q[10];
rz(-0.2391688685485374) q[10];
ry(0.0012178094063733562) q[11];
rz(-1.5539365059705652) q[11];
ry(2.7472172792732152) q[12];
rz(0.007979906504649747) q[12];
ry(-3.1390378114371944) q[13];
rz(1.1326301354132724) q[13];
ry(-0.0456010855256088) q[14];
rz(0.2671385194059797) q[14];
ry(-2.4141279661844957) q[15];
rz(-2.6967366443711307) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.327768495849403) q[0];
rz(0.17127465950167198) q[0];
ry(2.7947270014121806) q[1];
rz(-0.1438645259908439) q[1];
ry(-0.6114300021535302) q[2];
rz(1.1334355981097675) q[2];
ry(-2.031277740702315) q[3];
rz(-1.2865160682202568) q[3];
ry(0.1739773878196944) q[4];
rz(2.311903248924484) q[4];
ry(1.5658491953935216) q[5];
rz(-1.8278307173108184) q[5];
ry(-1.5697768691949472) q[6];
rz(2.3006994028794896) q[6];
ry(-1.5708163893546951) q[7];
rz(1.5637016255080738) q[7];
ry(-3.137378162314125) q[8];
rz(-2.538240084345989) q[8];
ry(-0.0005167681308240191) q[9];
rz(-1.4597749640837707) q[9];
ry(-1.5855220231327989) q[10];
rz(3.1147582746040645) q[10];
ry(2.8912616046965947) q[11];
rz(-3.109713991476792) q[11];
ry(2.4867138090563774) q[12];
rz(2.186033577590093) q[12];
ry(-3.1364028921373275) q[13];
rz(-2.776533283305777) q[13];
ry(-2.6647697469590232) q[14];
rz(-2.829628065875563) q[14];
ry(3.1094898159182645) q[15];
rz(1.540227931620172) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.8807679561760353) q[0];
rz(-3.0143141570758853) q[0];
ry(-0.3311278145168304) q[1];
rz(0.7724997776532803) q[1];
ry(-3.087020321024027) q[2];
rz(-0.8303750414844594) q[2];
ry(-0.08450139200116524) q[3];
rz(-2.099005232963199) q[3];
ry(-0.1705550451931446) q[4];
rz(0.6393196854233772) q[4];
ry(-0.30335557024154775) q[5];
rz(-2.6810749069984117) q[5];
ry(-3.021750692458872) q[6];
rz(-0.97996153076177) q[6];
ry(-1.3098747116450395) q[7];
rz(1.6996998754125918) q[7];
ry(-1.5703053700614673) q[8];
rz(-0.7350276658443408) q[8];
ry(1.574531628601803) q[9];
rz(-2.1499187473962316) q[9];
ry(-1.5711764851312164) q[10];
rz(0.026643001797181574) q[10];
ry(-1.5678282404678237) q[11];
rz(3.1352829471546677) q[11];
ry(0.010703912686938821) q[12];
rz(-2.1938649168115547) q[12];
ry(0.0005786768296737811) q[13];
rz(0.23105856675892983) q[13];
ry(1.55616626011035) q[14];
rz(-1.676863949724316) q[14];
ry(-1.8657398496492235) q[15];
rz(0.49347763830209895) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.2921562933407325) q[0];
rz(2.516583110426192) q[0];
ry(-1.3608772845522799) q[1];
rz(1.2758865601540252) q[1];
ry(-2.6109294771036424) q[2];
rz(0.4423742360218229) q[2];
ry(1.9812874182921192) q[3];
rz(-0.04183014090022397) q[3];
ry(1.560696958333546) q[4];
rz(1.6683333923508439) q[4];
ry(-1.6359919606671216) q[5];
rz(-2.62869625437711) q[5];
ry(-1.5711866249475497) q[6];
rz(-1.5713971767452346) q[6];
ry(-1.570287499957522) q[7];
rz(2.9535243895380177) q[7];
ry(0.0012449803110081295) q[8];
rz(2.6722774152716826) q[8];
ry(-3.135435137650406) q[9];
rz(2.764906723608898) q[9];
ry(1.5761466506628004) q[10];
rz(3.1164421382462373) q[10];
ry(1.5897269257459303) q[11];
rz(-2.8919113620570034) q[11];
ry(1.5733811641928268) q[12];
rz(1.5884762700786006) q[12];
ry(1.5329563841886904) q[13];
rz(-1.5736595214949585) q[13];
ry(0.9076671333376958) q[14];
rz(-2.9651087544244312) q[14];
ry(1.5016428877506334) q[15];
rz(-1.4567559556328087) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.6069732545706217) q[0];
rz(-2.2822279110914394) q[0];
ry(0.6624104804131665) q[1];
rz(-0.3031296606554754) q[1];
ry(-3.11900129625943) q[2];
rz(-0.2581638812526803) q[2];
ry(1.4980202199355521) q[3];
rz(-1.555373742273637) q[3];
ry(-1.3285226407010349) q[4];
rz(1.5136324346256185) q[4];
ry(0.13481023445742313) q[5];
rz(-2.842912943954324) q[5];
ry(1.8048698041117033) q[6];
rz(-1.5720408579129854) q[6];
ry(-3.1414385297492173) q[7];
rz(-0.19007050696450276) q[7];
ry(0.012310350570978736) q[8];
rz(-0.9307210671566511) q[8];
ry(3.1245796449121888) q[9];
rz(3.0068725269825043) q[9];
ry(0.08274476452461467) q[10];
rz(0.5449472667636447) q[10];
ry(-0.0021373891336438937) q[11];
rz(-2.5177006010311396) q[11];
ry(3.13545807153126) q[12];
rz(-1.5517555268781775) q[12];
ry(-3.047548835101978) q[13];
rz(-1.5766475637373338) q[13];
ry(-1.5695801131729068) q[14];
rz(0.6585420325354638) q[14];
ry(-1.5652168825847355) q[15];
rz(2.1030639974219185) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.6681967060113108) q[0];
rz(-2.688136431147368) q[0];
ry(1.5684067724859005) q[1];
rz(2.069033064193015) q[1];
ry(1.576572635239545) q[2];
rz(-3.130813239674569) q[2];
ry(1.6687319137300376) q[3];
rz(-1.2869850173074766) q[3];
ry(3.1269052857850754) q[4];
rz(-1.9147673539547432) q[4];
ry(-0.010051635004490998) q[5];
rz(0.7259320978013388) q[5];
ry(1.572060628473926) q[6];
rz(3.044849986768325) q[6];
ry(-1.570743037938931) q[7];
rz(-0.033699239191866535) q[7];
ry(-0.0002070707254597792) q[8];
rz(1.1563806057864543) q[8];
ry(-0.0069091569156148145) q[9];
rz(-1.8503802584550042) q[9];
ry(-0.05435583346084183) q[10];
rz(-1.2614449952504856) q[10];
ry(3.1344928904222438) q[11];
rz(-2.8671343480648974) q[11];
ry(1.5714683135137788) q[12];
rz(-2.2233408868370117) q[12];
ry(-1.523511957633105) q[13];
rz(-2.5471507636918966) q[13];
ry(3.141330822772565) q[14];
rz(-0.10946444796836115) q[14];
ry(-3.138664890882473) q[15];
rz(2.760542800554798) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.575606569811877) q[0];
rz(0.5656311409589031) q[0];
ry(-0.35436942270306204) q[1];
rz(-2.428142770286382) q[1];
ry(2.050415608470302) q[2];
rz(0.5603599265425885) q[2];
ry(1.7185369096257332) q[3];
rz(2.827194362957346) q[3];
ry(-2.0174832469894097) q[4];
rz(-2.8390505930585803) q[4];
ry(2.070058111256756) q[5];
rz(0.5854401580516244) q[5];
ry(-1.1303033263495026) q[6];
rz(-1.466105427905661) q[6];
ry(-2.7130576314958357) q[7];
rz(0.8676190514801272) q[7];
ry(-2.217629283090547) q[8];
rz(3.1311582279356007) q[8];
ry(-2.2853333836722665) q[9];
rz(-0.8503855514412404) q[9];
ry(2.0892653221268014) q[10];
rz(-1.9603557702776129) q[10];
ry(0.9163871496027456) q[11];
rz(3.1376151536903945) q[11];
ry(2.190939836223933) q[12];
rz(2.338326964149869) q[12];
ry(-2.234407698819284) q[13];
rz(3.1204098805530083) q[13];
ry(0.7895568212184272) q[14];
rz(-2.9233784851988016) q[14];
ry(0.805302290232908) q[15];
rz(-0.9109203368940683) q[15];