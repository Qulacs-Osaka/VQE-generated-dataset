OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.4036522475335413) q[0];
ry(2.7275997692390113) q[1];
cx q[0],q[1];
ry(0.3876200031890322) q[0];
ry(2.303564871920569) q[1];
cx q[0],q[1];
ry(-2.1907176790336154) q[1];
ry(-0.4021525809457805) q[2];
cx q[1],q[2];
ry(-0.8456853562982367) q[1];
ry(1.7567001128546538) q[2];
cx q[1],q[2];
ry(-2.309275319038552) q[2];
ry(-1.4113111324299545) q[3];
cx q[2],q[3];
ry(1.0617094409743275) q[2];
ry(0.1308329733988023) q[3];
cx q[2],q[3];
ry(2.3309202045999893) q[0];
ry(0.24576869520355785) q[1];
cx q[0],q[1];
ry(0.37871609870452544) q[0];
ry(-0.8242312174264099) q[1];
cx q[0],q[1];
ry(-0.9955475039516051) q[1];
ry(-2.169509430067876) q[2];
cx q[1],q[2];
ry(-2.9378386571667163) q[1];
ry(-2.402896884207347) q[2];
cx q[1],q[2];
ry(-0.6466161441835354) q[2];
ry(-1.6156868369049286) q[3];
cx q[2],q[3];
ry(0.36005453030719053) q[2];
ry(2.905254324187833) q[3];
cx q[2],q[3];
ry(1.75685048910126) q[0];
ry(-2.710784673273511) q[1];
cx q[0],q[1];
ry(-2.584244580155974) q[0];
ry(0.5088166900988966) q[1];
cx q[0],q[1];
ry(2.995590758117656) q[1];
ry(-0.9953243526701654) q[2];
cx q[1],q[2];
ry(-2.668163457815819) q[1];
ry(2.345106679751287) q[2];
cx q[1],q[2];
ry(-0.48350990567912816) q[2];
ry(-1.7953150786415293) q[3];
cx q[2],q[3];
ry(-1.4329041638427606) q[2];
ry(0.7522538257177035) q[3];
cx q[2],q[3];
ry(-2.9878776936783273) q[0];
ry(-2.4769338364098155) q[1];
cx q[0],q[1];
ry(-2.5458219183455397) q[0];
ry(1.1776020753193817) q[1];
cx q[0],q[1];
ry(-2.21575358896456) q[1];
ry(-1.2455323746995184) q[2];
cx q[1],q[2];
ry(2.0753341448618077) q[1];
ry(-2.125922248591861) q[2];
cx q[1],q[2];
ry(2.007063574448479) q[2];
ry(-2.0907852344138167) q[3];
cx q[2],q[3];
ry(2.663939275903992) q[2];
ry(2.4497725146005247) q[3];
cx q[2],q[3];
ry(1.154159939685353) q[0];
ry(-2.362211032005657) q[1];
cx q[0],q[1];
ry(-0.42236507605689066) q[0];
ry(-3.0952818307799848) q[1];
cx q[0],q[1];
ry(-0.4322631295449109) q[1];
ry(-1.9335026475149546) q[2];
cx q[1],q[2];
ry(1.119718291244073) q[1];
ry(-0.12745155676871178) q[2];
cx q[1],q[2];
ry(-1.3412292914418291) q[2];
ry(2.7977569938317104) q[3];
cx q[2],q[3];
ry(3.074957375162518) q[2];
ry(-2.287759064255614) q[3];
cx q[2],q[3];
ry(-1.9104048019495448) q[0];
ry(2.0062468532033284) q[1];
cx q[0],q[1];
ry(1.8217835322177305) q[0];
ry(-1.5531287931358848) q[1];
cx q[0],q[1];
ry(0.16818712211827072) q[1];
ry(-3.1379016410683316) q[2];
cx q[1],q[2];
ry(-1.2727638531075351) q[1];
ry(2.6464779093474635) q[2];
cx q[1],q[2];
ry(-1.2575236784291013) q[2];
ry(-0.2250090593842229) q[3];
cx q[2],q[3];
ry(-0.8258841015186382) q[2];
ry(2.099441356860667) q[3];
cx q[2],q[3];
ry(0.20073763389873514) q[0];
ry(-2.0339540012895014) q[1];
cx q[0],q[1];
ry(1.1850115435185213) q[0];
ry(1.6377530215817604) q[1];
cx q[0],q[1];
ry(1.842871160663429) q[1];
ry(-2.7253955063463318) q[2];
cx q[1],q[2];
ry(2.2212585765049693) q[1];
ry(-3.098580279872669) q[2];
cx q[1],q[2];
ry(-0.542843151058757) q[2];
ry(2.879892016750854) q[3];
cx q[2],q[3];
ry(-0.8302346638246316) q[2];
ry(0.7336422687065784) q[3];
cx q[2],q[3];
ry(-2.9748064457942296) q[0];
ry(0.006587444552076408) q[1];
cx q[0],q[1];
ry(-0.18177802200632698) q[0];
ry(-0.613425298078881) q[1];
cx q[0],q[1];
ry(-2.1422798175155293) q[1];
ry(-0.6788354950236491) q[2];
cx q[1],q[2];
ry(1.5387667035569588) q[1];
ry(-1.9787139142315953) q[2];
cx q[1],q[2];
ry(-2.5447353818702068) q[2];
ry(-1.0014446938443529) q[3];
cx q[2],q[3];
ry(2.004332836381254) q[2];
ry(-2.2675874945622594) q[3];
cx q[2],q[3];
ry(-0.6673660246321029) q[0];
ry(-0.7863383470659382) q[1];
cx q[0],q[1];
ry(0.29178767066938754) q[0];
ry(1.967772703715629) q[1];
cx q[0],q[1];
ry(-2.8161862526568138) q[1];
ry(-1.115489047204722) q[2];
cx q[1],q[2];
ry(1.048469469776208) q[1];
ry(0.16032401784167852) q[2];
cx q[1],q[2];
ry(1.1308423302680124) q[2];
ry(-2.1702381086448295) q[3];
cx q[2],q[3];
ry(-2.4808562443749262) q[2];
ry(1.3515711522170974) q[3];
cx q[2],q[3];
ry(0.0781366128465917) q[0];
ry(1.9841216689800025) q[1];
cx q[0],q[1];
ry(-0.8286948765939461) q[0];
ry(-1.2880833109489753) q[1];
cx q[0],q[1];
ry(2.05293465302875) q[1];
ry(3.085744582267002) q[2];
cx q[1],q[2];
ry(-0.890152832251558) q[1];
ry(2.0776642280247346) q[2];
cx q[1],q[2];
ry(1.1354012500575275) q[2];
ry(-1.0573900582878046) q[3];
cx q[2],q[3];
ry(0.2461658210069988) q[2];
ry(-2.5699684055405476) q[3];
cx q[2],q[3];
ry(2.5811506558762876) q[0];
ry(-1.520776761156673) q[1];
ry(2.88491998897854) q[2];
ry(1.1990500686286474) q[3];