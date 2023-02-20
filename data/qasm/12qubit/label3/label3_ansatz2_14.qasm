OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.1409951736328194) q[0];
rz(2.15292986692033) q[0];
ry(-0.0009915798532826477) q[1];
rz(-2.366538337343704) q[1];
ry(3.1221337747149116) q[2];
rz(1.3148937969061436) q[2];
ry(-0.14242362996738817) q[3];
rz(-1.730858372806357) q[3];
ry(-1.626490150411398) q[4];
rz(-1.577915455766295) q[4];
ry(-1.8119841717787808) q[5];
rz(2.3726676217428913) q[5];
ry(3.138992748750148) q[6];
rz(1.4553826756418353) q[6];
ry(1.5710815001603409) q[7];
rz(-1.0282778711231595) q[7];
ry(3.105963554627992) q[8];
rz(1.8556858203526787) q[8];
ry(1.0587099851951827) q[9];
rz(1.3026530646682095) q[9];
ry(-0.0002390709039974044) q[10];
rz(-2.1955488294410763) q[10];
ry(-0.06354291534906054) q[11];
rz(-1.7983814034079422) q[11];
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
ry(1.8890532916593372e-05) q[0];
rz(-2.1973568992278167) q[0];
ry(-3.1394740145901894) q[1];
rz(-1.9795815960999201) q[1];
ry(-0.00024554787542550264) q[2];
rz(1.7702878157713686) q[2];
ry(0.0027645060142391896) q[3];
rz(1.7129951126857923) q[3];
ry(-2.7222766258307316) q[4];
rz(2.784413514104556) q[4];
ry(3.1409349420848187) q[5];
rz(0.8064413165222621) q[5];
ry(-0.0011206462404791393) q[6];
rz(1.9817913983535256) q[6];
ry(1.5629844562466497e-05) q[7];
rz(2.6236247271269075) q[7];
ry(0.0023034593583934893) q[8];
rz(2.8092086235831557) q[8];
ry(3.129690612169638) q[9];
rz(-1.8566963057121546) q[9];
ry(-3.141465082032256) q[10];
rz(-1.331358594558712) q[10];
ry(-3.1376899462970997) q[11];
rz(-1.8021925314588538) q[11];
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
ry(-3.140527735475681) q[0];
rz(1.398673751794435) q[0];
ry(-3.1406590128036735) q[1];
rz(-2.288049213291367) q[1];
ry(0.012043159205775743) q[2];
rz(-3.1166097801435093) q[2];
ry(-3.071790901291927) q[3];
rz(3.063277510050641) q[3];
ry(3.133566986533861) q[4];
rz(1.2722225018974536) q[4];
ry(1.5911387091933191) q[5];
rz(2.7543328824931255) q[5];
ry(3.138910958452861) q[6];
rz(0.1155088923274441) q[6];
ry(1.5426747261892069) q[7];
rz(1.7745973848177838) q[7];
ry(1.56959387523536) q[8];
rz(-0.01932631747358382) q[8];
ry(0.50674256459739) q[9];
rz(-0.2530675154453909) q[9];
ry(-1.5708281467912713) q[10];
rz(3.905270014317646e-05) q[10];
ry(1.49517594932645) q[11];
rz(-3.062189762357509) q[11];
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
ry(1.5009154488933791) q[0];
rz(2.925424656753144) q[0];
ry(3.14141934578804) q[1];
rz(-2.950612656306244) q[1];
ry(1.5709845502138273) q[2];
rz(-0.0016113277427720618) q[2];
ry(3.138168348935144) q[3];
rz(0.005250637459492502) q[3];
ry(-0.0029802497425805478) q[4];
rz(-0.3855985711730306) q[4];
ry(-6.4874605949978625e-06) q[5];
rz(0.00731471638568697) q[5];
ry(0.43651082940579405) q[6];
rz(0.38955996858757036) q[6];
ry(0.0009111545562934095) q[7];
rz(2.938461068808159) q[7];
ry(0.06812228492004824) q[8];
rz(-1.6989988510600638) q[8];
ry(0.00022724501412252351) q[9];
rz(1.8395599055015186) q[9];
ry(1.6359552957471213) q[10];
rz(-3.065156455073662) q[10];
ry(-3.1392344996764505) q[11];
rz(-1.4919888601549178) q[11];
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
ry(-0.005235298750279031) q[0];
rz(0.2459267463045549) q[0];
ry(-0.0016912366478694096) q[1];
rz(2.672984702100569) q[1];
ry(1.5752099643057191) q[2];
rz(-3.108772513490742) q[2];
ry(-3.1211767221166116) q[3];
rz(1.834277160136149) q[3];
ry(0.024686206216339105) q[4];
rz(-2.8123557077145676) q[4];
ry(-3.140892124197735) q[5];
rz(0.4366099919672104) q[5];
ry(-3.1415404752451654) q[6];
rz(-0.3903617889519589) q[6];
ry(-2.25414274509121) q[7];
rz(0.8144069341452588) q[7];
ry(-0.0021347254492454226) q[8];
rz(-1.990949401007887) q[8];
ry(1.5697274705912738) q[9];
rz(1.3663766187008364) q[9];
ry(-3.1410196360756606) q[10];
rz(0.26654868152344857) q[10];
ry(1.5773358239926498) q[11];
rz(1.5735480708429268) q[11];
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
ry(0.7916449549637764) q[0];
rz(1.9274411701613872) q[0];
ry(-1.5687766762739122) q[1];
rz(-3.1390916412420755) q[1];
ry(-2.359632528406764) q[2];
rz(2.040693163104037) q[2];
ry(-1.5807498787219754) q[3];
rz(3.095761628118594) q[3];
ry(3.12159756903456) q[4];
rz(2.960791931898971) q[4];
ry(0.3190553096498583) q[5];
rz(-1.3115683546462586) q[5];
ry(1.5514209992448322) q[6];
rz(-3.1048798755548304) q[6];
ry(3.132899841774762) q[7];
rz(2.617494634462775) q[7];
ry(1.5647982837803456) q[8];
rz(-3.1080742099696725) q[8];
ry(2.7297809983415338) q[9];
rz(2.726684270429749) q[9];
ry(-0.019875581566477507) q[10];
rz(-2.257960839464473) q[10];
ry(-1.4546075433585584) q[11];
rz(-1.6155954268253172) q[11];
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
ry(-3.0692761643829116) q[0];
rz(-2.743506527044469) q[0];
ry(1.2390138804246003) q[1];
rz(1.5700386149787509) q[1];
ry(-0.06614414840218519) q[2];
rz(-0.42671541984267486) q[2];
ry(-1.5596986766842909) q[3];
rz(-0.001965418092710003) q[3];
ry(0.0001535427579876387) q[4];
rz(-2.9954468969974184) q[4];
ry(3.141191192787293) q[5];
rz(1.4283026681556423) q[5];
ry(-0.3209735613635454) q[6];
rz(1.554211607664705) q[6];
ry(3.1412859252727934) q[7];
rz(-1.3344450680647664) q[7];
ry(-3.1413237384455694) q[8];
rz(1.8531493500656468) q[8];
ry(3.141400603247005) q[9];
rz(-0.19059968655277704) q[9];
ry(3.138938239611621) q[10];
rz(-2.3965008445794247) q[10];
ry(-3.138799293631091) q[11];
rz(1.5187536621233297) q[11];
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
ry(-1.5703447571916995) q[0];
rz(-1.1859262966212882) q[0];
ry(-0.5946003825790944) q[1];
rz(2.0216472017794986) q[1];
ry(-1.5737122572413162) q[2];
rz(1.5773958410051927) q[2];
ry(-1.5738238524881563) q[3];
rz(-0.1984772409427782) q[3];
ry(0.05079733305416578) q[4];
rz(-1.5391045157537946) q[4];
ry(-1.8132769894850427) q[5];
rz(-1.6318481386523054) q[5];
ry(-0.7909535475349578) q[6];
rz(-0.25524430132525) q[6];
ry(-1.1567785636427736) q[7];
rz(-1.5823794272279679) q[7];
ry(-2.3404843596445684) q[8];
rz(-2.7512318691054234) q[8];
ry(2.006773836406494) q[9];
rz(-2.9753752260171535) q[9];
ry(-3.1304636972122424) q[10];
rz(1.2427286648349594) q[10];
ry(0.27930808994721146) q[11];
rz(0.0721419707506943) q[11];
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
ry(3.141337474108273) q[0];
rz(-2.054370162589919) q[0];
ry(-3.140710336626829) q[1];
rz(1.846752246578751) q[1];
ry(-0.00019711261786525824) q[2];
rz(1.0916707998005597) q[2];
ry(-1.5699775405476846) q[3];
rz(-1.4044972982544945) q[3];
ry(2.977439672058828) q[4];
rz(-2.6444836139162664) q[4];
ry(-1.049674500579102) q[5];
rz(1.595833677621071) q[5];
ry(-0.0005624696428412206) q[6];
rz(2.3407567455183216) q[6];
ry(-0.25324630834377165) q[7];
rz(2.9852850487437284) q[7];
ry(-2.973727540006233) q[8];
rz(-0.3779125376667167) q[8];
ry(-1.7552688900976179) q[9];
rz(-2.5566304853788875) q[9];
ry(-2.7079871238950486) q[10];
rz(2.4145802972639765) q[10];
ry(1.5641172871724587) q[11];
rz(-1.5338871394891878) q[11];
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
ry(-3.140266768735249) q[0];
rz(-0.48996580891030833) q[0];
ry(3.1404491027581987) q[1];
rz(0.345511257547689) q[1];
ry(3.1401847771279647) q[2];
rz(-2.344629539580853) q[2];
ry(-3.1339298408944565) q[3];
rz(0.16767721272516134) q[3];
ry(-1.2204563005059584e-05) q[4];
rz(-0.4950911013953024) q[4];
ry(-1.5706678109497194) q[5];
rz(0.00023940605091449493) q[5];
ry(-0.0015055483141570094) q[6];
rz(0.9580381114531994) q[6];
ry(3.1414221234380535) q[7];
rz(-1.7313174023011284) q[7];
ry(-3.1415616978180037) q[8];
rz(2.6657564240962377) q[8];
ry(1.5709327278275147) q[9];
rz(-0.0015504182960516426) q[9];
ry(-3.1415230325416243) q[10];
rz(-0.7279967413500855) q[10];
ry(1.5716432613003137) q[11];
rz(0.04862212592873479) q[11];
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
ry(-0.0002643416975747521) q[0];
rz(-2.778811906611957) q[0];
ry(0.00011393760479183611) q[1];
rz(-0.5190510137215262) q[1];
ry(3.1414330741045884) q[2];
rz(-2.6403881487857586) q[2];
ry(-1.5676453125772318) q[3];
rz(2.9105460366988614) q[3];
ry(1.5588157040753616) q[4];
rz(1.5708135482732564) q[4];
ry(1.5812530441726498) q[5];
rz(-1.5692367198077346) q[5];
ry(3.1154449215565867) q[6];
rz(-1.5692377833491233) q[6];
ry(1.5708659700983958) q[7];
rz(-1.5668310542769526) q[7];
ry(-0.002496538424461261) q[8];
rz(-2.0744683879536976) q[8];
ry(1.527501019096929) q[9];
rz(-1.2850479488397042) q[9];
ry(1.5601927340266368) q[10];
rz(-1.5962903023801946) q[10];
ry(0.03819481215256877) q[11];
rz(1.521179495969105) q[11];
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
ry(3.122765688995112) q[0];
rz(2.214265638964007) q[0];
ry(-1.5665126736430934) q[1];
rz(0.3720595636731945) q[1];
ry(0.01694420816739051) q[2];
rz(-1.3883098811378272) q[2];
ry(-1.5726858257191738) q[3];
rz(1.8220823574273362) q[3];
ry(1.5828484762258421) q[4];
rz(1.8811965943736164) q[4];
ry(-2.87977797549202) q[5];
rz(3.1411803730142833) q[5];
ry(-0.03211494823877459) q[6];
rz(-3.13924738017899) q[6];
ry(1.5727530224660295) q[7];
rz(1.4100333171056132) q[7];
ry(-0.00024911333512012135) q[8];
rz(0.3832405910363761) q[8];
ry(-3.1406194676317067) q[9];
rz(-2.8452109638320766) q[9];
ry(-0.012492055319767435) q[10];
rz(-0.10113312278568327) q[10];
ry(0.3036181140194439) q[11];
rz(3.1316478877659475) q[11];
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
ry(-3.141267286127792) q[0];
rz(-1.1716785803454932) q[0];
ry(-7.675300096411727e-05) q[1];
rz(-0.3704767642869733) q[1];
ry(-3.141036345986063) q[2];
rz(-0.24081238675589223) q[2];
ry(3.1384599381691083) q[3];
rz(1.8228878504194395) q[3];
ry(3.13967398444747) q[4];
rz(-1.2626509894032005) q[4];
ry(1.570440900924658) q[5];
rz(0.000592921791060519) q[5];
ry(-3.1375168314651742) q[6];
rz(0.07602290270587808) q[6];
ry(0.010368917594381532) q[7];
rz(1.7299053696006974) q[7];
ry(3.1415054519416246) q[8];
rz(2.1263838656891014) q[8];
ry(1.569691790730263) q[9];
rz(0.018487329436243125) q[9];
ry(3.1402083931917377) q[10];
rz(-0.13216878034387491) q[10];
ry(-1.569907477252797) q[11];
rz(-0.005953479197919443) q[11];
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
ry(3.1350638775380846) q[0];
rz(-0.264342929594996) q[0];
ry(-2.5295019119103053) q[1];
rz(1.5719119767049192) q[1];
ry(-0.01894822988324929) q[2];
rz(0.9328218343833643) q[2];
ry(-1.1351582307048569) q[3];
rz(2.7419755251965765) q[3];
ry(1.5462321475385634) q[4];
rz(2.8333697592415015) q[4];
ry(-2.0933237761283365) q[5];
rz(-1.3789159734664977) q[5];
ry(1.5713490679334505) q[6];
rz(1.1828467970879668) q[6];
ry(-1.0253256756331766) q[7];
rz(-1.6417999243290566) q[7];
ry(-0.0005976215941689844) q[8];
rz(2.5554598189391062) q[8];
ry(0.14391488544761472) q[9];
rz(1.600272565196275) q[9];
ry(3.123318500246361) q[10];
rz(0.6678532201373982) q[10];
ry(0.31025621282155535) q[11];
rz(1.4591690118660772) q[11];
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
ry(-3.1275832045506924) q[0];
rz(-1.535975916812009) q[0];
ry(-1.5845385696545236) q[1];
rz(1.6687823205488324) q[1];
ry(-0.0008104943519837704) q[2];
rz(2.6476243258404013) q[2];
ry(5.771026310265906e-05) q[3];
rz(-2.716270304577336) q[3];
ry(1.5664216897370107) q[4];
rz(-1.1401851363622404) q[4];
ry(3.1254331860710733) q[5];
rz(2.079730471209616) q[5];
ry(6.0525184850800866e-05) q[6];
rz(2.230069389683715) q[6];
ry(3.1279544342822945) q[7];
rz(-1.7074931215875386) q[7];
ry(1.5753564651834753) q[8];
rz(-0.018518689484527506) q[8];
ry(3.123359571495741) q[9];
rz(1.5719275688139733) q[9];
ry(-0.003877083279868821) q[10];
rz(2.4674821399978732) q[10];
ry(3.1237284756181425) q[11];
rz(-1.6091986081013567) q[11];
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
ry(-2.46238476840567) q[0];
rz(-2.370541418389578) q[0];
ry(-3.1022307740362076) q[1];
rz(0.8283303281996145) q[1];
ry(2.151610956102779) q[2];
rz(1.121632321585996) q[2];
ry(-3.1312145081226306) q[3];
rz(2.850606089200333) q[3];
ry(0.19108379872952816) q[4];
rz(1.1324203108003246) q[4];
ry(-3.1139690007639245) q[5];
rz(0.1933254560104709) q[5];
ry(-0.0008936207857530576) q[6];
rz(-1.8339097322354883) q[6];
ry(-0.09338788267926325) q[7];
rz(-3.0796542237501536) q[7];
ry(-0.22759631817492032) q[8];
rz(1.7356276137734048) q[8];
ry(-3.089263263113828) q[9];
rz(2.910322091195852) q[9];
ry(1.5659144784275782) q[10];
rz(-1.703857864111346) q[10];
ry(-0.054045606204679686) q[11];
rz(-2.2400802478307145) q[11];
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
ry(0.00194735171695104) q[0];
rz(2.3823403894613975) q[0];
ry(3.1385059225761003) q[1];
rz(0.1935681225072345) q[1];
ry(0.0015785378494862312) q[2];
rz(-1.1120717152621955) q[2];
ry(-3.141572239951118) q[3];
rz(-0.3196148601069418) q[3];
ry(1.5701485296506847) q[4];
rz(-1.6093086193742612) q[4];
ry(3.138073378842447) q[5];
rz(0.6667585295343395) q[5];
ry(1.5715321606824735) q[6];
rz(3.140331473237622) q[6];
ry(-3.14006778460339) q[7];
rz(2.7841653092402727) q[7];
ry(3.1411481542165425) q[8];
rz(-2.4105175356617012) q[8];
ry(3.141284356014147) q[9];
rz(-0.49093354514777937) q[9];
ry(0.027060492099779232) q[10];
rz(-1.4367852466701212) q[10];
ry(-3.141457477553076) q[11];
rz(1.402078356477227) q[11];
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
ry(1.5707347446776707) q[0];
rz(-2.2790640148432564) q[0];
ry(-3.1343763717555517) q[1];
rz(-2.831819940848367) q[1];
ry(-1.5706639770228223) q[2];
rz(0.8628152386522263) q[2];
ry(-1.5641496848587815) q[3];
rz(-2.294554673607096) q[3];
ry(-0.005295355774434185) q[4];
rz(-0.6691931525888825) q[4];
ry(-0.00039140736369258633) q[5];
rz(-3.086432266231601) q[5];
ry(-1.5749255430861728) q[6];
rz(-0.8814145234777361) q[6];
ry(0.0015011291595535853) q[7];
rz(1.2002438733063414) q[7];
ry(-0.0019142931343250071) q[8];
rz(-1.296612052122089) q[8];
ry(-0.002124795674574464) q[9];
rz(1.1520129844376532) q[9];
ry(-1.5730153271336214) q[10];
rz(-0.7108962563484224) q[10];
ry(0.0004576417969097691) q[11];
rz(-2.7176576328572186) q[11];