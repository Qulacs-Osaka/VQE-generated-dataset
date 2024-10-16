OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.874975711005797) q[0];
rz(1.3492083096109897) q[0];
ry(0.07107114085333134) q[1];
rz(-1.5480286663657292) q[1];
ry(-3.1415888377035373) q[2];
rz(0.30128350713126545) q[2];
ry(1.570791325657716) q[3];
rz(2.9685604497027107) q[3];
ry(0.4395277764387808) q[4];
rz(3.1412781432982415) q[4];
ry(0.0005386838758506066) q[5];
rz(0.14976451924177378) q[5];
ry(-3.1415748958705465) q[6];
rz(2.7480032972556874) q[6];
ry(-3.141127253668848) q[7];
rz(0.4031279953498646) q[7];
ry(-1.5707745336889822) q[8];
rz(-3.141534253451919) q[8];
ry(1.5627727690265754) q[9];
rz(1.5760649384449914) q[9];
ry(-1.570807259924093) q[10];
rz(3.141500375500993) q[10];
ry(-6.729817356601884e-05) q[11];
rz(-1.364531840877743) q[11];
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
ry(1.2218396813080319) q[0];
rz(0.07677765148051799) q[0];
ry(1.5304635067230148) q[1];
rz(-2.1474707407165434) q[1];
ry(3.1415883740288026) q[2];
rz(1.5858854510812608) q[2];
ry(-3.1412208373621655) q[3];
rz(2.9685154732486008) q[3];
ry(-1.5709471503834243) q[4];
rz(1.6867989471069065) q[4];
ry(-1.5707754107227594) q[5];
rz(6.144071811942098e-05) q[5];
ry(-1.5708302461043224) q[6];
rz(-1.5707405599664264) q[6];
ry(-0.014903378780254606) q[7];
rz(1.3987859938705924) q[7];
ry(-1.6619729594762536) q[8];
rz(-2.5138257825833525) q[8];
ry(1.526383684819071) q[9];
rz(-1.389800802999357) q[9];
ry(-2.93025334527039) q[10];
rz(1.570701600511165) q[10];
ry(-1.5690646289386239) q[11];
rz(-2.396932919998183) q[11];
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
ry(1.570801844830565) q[0];
rz(-1.5802831934379962) q[0];
ry(3.14158084962428) q[1];
rz(-0.575612165744573) q[1];
ry(1.5707828039918879) q[2];
rz(-0.1888384386077261) q[2];
ry(-1.5706827338930935) q[3];
rz(-1.3756793665461255) q[3];
ry(3.6288987478344146e-06) q[4];
rz(0.3942662270923538) q[4];
ry(-0.36140049592961465) q[5];
rz(1.5711558449010627) q[5];
ry(1.5707770905399663) q[6];
rz(2.553329275673396) q[6];
ry(-1.5708827678031287) q[7];
rz(-2.90085985031664) q[7];
ry(-3.141508707140668) q[8];
rz(-1.798691539774099) q[8];
ry(3.141586130978092) q[9];
rz(-1.7544743540251633) q[9];
ry(-1.7658680682864203) q[10];
rz(-1.5708110914368778) q[10];
ry(8.190680683028972e-06) q[11];
rz(-2.3152455645829306) q[11];
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
ry(-0.005526711657543353) q[0];
rz(1.5803209811472128) q[0];
ry(1.3871371272280673) q[1];
rz(1.5618599047247044) q[1];
ry(-7.366535304448859e-05) q[2];
rz(-1.382055376776047) q[2];
ry(1.5708056609496308) q[3];
rz(-1.570949027534322) q[3];
ry(2.714202952573757) q[4];
rz(-1.1858124934320449) q[4];
ry(0.4126624812854267) q[5];
rz(-1.5627812032254251) q[5];
ry(-3.141583168433539) q[6];
rz(1.7531245982904409) q[6];
ry(-8.42855931605049e-05) q[7];
rz(1.7311175479050709) q[7];
ry(-9.270769316493954e-05) q[8];
rz(-0.7132836991853395) q[8];
ry(9.48484944154986e-06) q[9];
rz(-1.0862596530290005) q[9];
ry(-1.0901594389494687) q[10];
rz(2.3079078965544787e-05) q[10];
ry(2.9536056933803345) q[11];
rz(-0.5690090852290024) q[11];
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
ry(1.572434183231493) q[0];
rz(-2.750001571020631) q[0];
ry(3.1411326456531596) q[1];
rz(-1.6562898895587568) q[1];
ry(-1.5707667323086154) q[2];
rz(-1.5709014693393633) q[2];
ry(-1.5707947906437942) q[3];
rz(-3.14158208669784) q[3];
ry(-0.00013379236067923017) q[4];
rz(1.307360143265515) q[4];
ry(1.5695734151474838) q[5];
rz(1.5713045501556624) q[5];
ry(-6.914174161050521e-06) q[6];
rz(-2.7027839570476875) q[6];
ry(-1.861908811519253e-06) q[7];
rz(2.740577144050844) q[7];
ry(3.141539917309706) q[8];
rz(2.004866725255134) q[8];
ry(-1.5707953312538903) q[9];
rz(-2.6660675549448274) q[9];
ry(-1.5707893404794133) q[10];
rz(2.6815969107182887) q[10];
ry(-3.709672534700168e-08) q[11];
rz(0.5691510003874226) q[11];
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
ry(-1.0253683946359615e-05) q[0];
rz(-1.1359485435345862) q[0];
ry(3.140467531772884) q[1];
rz(-2.3310945226878026) q[1];
ry(-0.5094183281546565) q[2];
rz(0.00010136506502923673) q[2];
ry(-1.5705516374209454) q[3];
rz(-1.5707941126323468) q[3];
ry(-1.5704994412224893) q[4];
rz(-1.987057892872157) q[4];
ry(-2.8753877490390516) q[5];
rz(-1.5703777547484297) q[5];
ry(1.5708128594313167) q[6];
rz(-1.5707800356846457) q[6];
ry(-1.570756348297965) q[7];
rz(-2.632132369182353) q[7];
ry(5.83346726729224e-05) q[8];
rz(-1.6861638164891088) q[8];
ry(2.4095926854493378e-05) q[9];
rz(2.5869243380444686) q[9];
ry(-3.1415186272399445) q[10];
rz(2.66050089702255) q[10];
ry(1.3031872244613203) q[11];
rz(-1.5691631319465393) q[11];
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
ry(-3.1415331802218374) q[0];
rz(-0.2634004045676222) q[0];
ry(2.255113322569554e-06) q[1];
rz(-2.9020370537521014) q[1];
ry(-1.5707923146152603) q[2];
rz(0.6692005255180401) q[2];
ry(1.5708090683533058) q[3];
rz(-1.6884785820635972) q[3];
ry(5.4476400293649523e-05) q[4];
rz(-1.1544439054814664) q[4];
ry(-1.7417421511270286) q[5];
rz(-0.09896922241676845) q[5];
ry(-1.5707907467952282) q[6];
rz(1.0625987834104513) q[6];
ry(-1.5708387813632831) q[7];
rz(1.5708474289659107) q[7];
ry(-1.9215089616595608e-06) q[8];
rz(1.3443670657142466) q[8];
ry(-8.430564028216736e-06) q[9];
rz(1.6650788651596682) q[9];
ry(3.1414359079777108) q[10];
rz(-0.22078667517958428) q[10];
ry(-1.3998601901837198) q[11];
rz(1.570489828041475) q[11];
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
ry(3.14158763025642) q[0];
rz(2.051774140280836) q[0];
ry(3.0844361185313214) q[1];
rz(2.6900427245822853) q[1];
ry(3.1415683980054228) q[2];
rz(2.3962512581395585) q[2];
ry(-2.332501433066625e-05) q[3];
rz(-1.4531631072720597) q[3];
ry(1.5711334759737863) q[4];
rz(3.14155414932168) q[4];
ry(-1.6049998221608357) q[5];
rz(0.33189277142955553) q[5];
ry(4.733963004088792e-05) q[6];
rz(2.374266735658633) q[6];
ry(-1.5768539664132435) q[7];
rz(-3.141517275266902) q[7];
ry(0.00032623721841542317) q[8];
rz(1.4803585231036622) q[8];
ry(3.0289290582805166) q[9];
rz(1.585855601070856) q[9];
ry(0.00010408995736787878) q[10];
rz(-1.372766947896692) q[10];
ry(1.4207630825818134) q[11];
rz(-0.07596034584644468) q[11];
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
ry(-1.5707768997028602) q[0];
rz(-2.500169337865665) q[0];
ry(3.141063957786029) q[1];
rz(1.3850711648819243) q[1];
ry(3.1415628024391653) q[2];
rz(0.15626287954147866) q[2];
ry(1.5705839403074064) q[3];
rz(1.365624834411797) q[3];
ry(-1.5707936176858404) q[4];
rz(1.5772795920566753) q[4];
ry(-1.570789634205659) q[5];
rz(-0.615133711522823) q[5];
ry(-3.141474463991057) q[6];
rz(0.5048199145740183) q[6];
ry(1.5708563600239192) q[7];
rz(-3.141489573966928) q[7];
ry(1.5707840525790937) q[8];
rz(-0.7062721615677363) q[8];
ry(1.5707887460020737) q[9];
rz(-3.141074574009681) q[9];
ry(-1.5708016192086545) q[10];
rz(-1.4054525669995407e-05) q[10];
ry(-3.1415868433390415) q[11];
rz(2.9840141013004886) q[11];
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
ry(2.154655812527208e-05) q[0];
rz(-2.212409399866445) q[0];
ry(3.141589797200452) q[1];
rz(-2.190125891950105) q[1];
ry(-1.914972625320468) q[2];
rz(0.032243935127904494) q[2];
ry(-5.7750621490271214e-05) q[3];
rz(1.7759358991840455) q[3];
ry(5.984704081640757e-06) q[4];
rz(3.1349078866564186) q[4];
ry(-6.138963254786195e-06) q[5];
rz(2.1859608087355027) q[5];
ry(3.14157782325039) q[6];
rz(0.20954196798896985) q[6];
ry(1.570802995077663) q[7];
rz(3.141590195770824) q[7];
ry(-6.949602451319157e-05) q[8];
rz(-2.435257916570581) q[8];
ry(0.08474179336373489) q[9];
rz(-1.7132623060303898) q[9];
ry(-0.7002450197229964) q[10];
rz(-1.7184950760926156) q[10];
ry(3.141541781538451) q[11];
rz(1.489212811869681) q[11];
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
ry(-1.570782223894395) q[0];
rz(2.547998027454489) q[0];
ry(-2.819901609056177e-05) q[1];
rz(-2.7012035244292476) q[1];
ry(3.1415826628059373) q[2];
rz(2.978073767670618) q[2];
ry(-1.5709653492137745) q[3];
rz(-8.738620364323484e-05) q[3];
ry(-1.5708391706626328) q[4];
rz(1.897924154066109e-05) q[4];
ry(1.570750430320427) q[5];
rz(0.00023163212427319583) q[5];
ry(1.5707780811664378) q[6];
rz(-2.987973348130563) q[6];
ry(-1.5707916167193898) q[7];
rz(-3.1415842014196906) q[7];
ry(1.570721150117189) q[8];
rz(0.873825464429704) q[8];
ry(-3.1415841724585922) q[9];
rz(-0.119276736851691) q[9];
ry(-3.141522854950318) q[10];
rz(2.8957145839986818) q[10];
ry(1.5706701721405718) q[11];
rz(-1.5710964356883794) q[11];
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
ry(5.337361649537797e-06) q[0];
rz(-2.5480043182757623) q[0];
ry(-1.570803404317579) q[1];
rz(2.344153704943771) q[1];
ry(1.4966387954957778e-05) q[2];
rz(0.19578073325365256) q[2];
ry(-1.5708000748871127) q[3];
rz(2.0937777808170877) q[3];
ry(-1.570786780328037) q[4];
rz(-3.141581667337356) q[4];
ry(1.5707970813973788) q[5];
rz(-1.4659227381308428e-05) q[5];
ry(-1.0490359557029052e-05) q[6];
rz(-0.15355542149374257) q[6];
ry(1.5707974835801135) q[7];
rz(3.1415246114383972) q[7];
ry(-3.141566641821631) q[8];
rz(-2.2677196067061116) q[8];
ry(3.141590231288089) q[9];
rz(1.5934733794690272) q[9];
ry(3.1415904360300093) q[10];
rz(1.4726265807849959) q[10];
ry(-1.5704891102193586) q[11];
rz(-2.7608919321739265) q[11];
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
ry(1.5707399424082293) q[0];
rz(2.2586034579302696) q[0];
ry(4.833318610078268e-06) q[1];
rz(-0.9506715500503171) q[1];
ry(-1.570793032145109) q[2];
rz(-0.8829661878827837) q[2];
ry(2.0910778991911627e-05) q[3];
rz(-2.271378125869953) q[3];
ry(1.570792249354537) q[4];
rz(0.6877946839701456) q[4];
ry(1.5707964399781789) q[5];
rz(-1.7483918333437387) q[5];
ry(1.5707927108158466) q[6];
rz(2.258615531088954) q[6];
ry(-1.5707966887360325) q[7];
rz(-0.1776023693732544) q[7];
ry(1.5707952727439847) q[8];
rz(2.2585968846872717) q[8];
ry(-1.5707982295938097) q[9];
rz(-0.17759097786542472) q[9];
ry(1.570802387122227) q[10];
rz(-2.4537784191633842) q[10];
ry(-3.141287278167777) q[11];
rz(-2.9384722468298707) q[11];