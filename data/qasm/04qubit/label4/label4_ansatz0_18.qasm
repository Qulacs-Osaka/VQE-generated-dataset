OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0469812243363666) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.00257536876496211) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08677705997502885) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11979143114748078) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.010059782402028046) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.005933732746983922) q[3];
cx q[2],q[3];
rx(-0.0466676718913384) q[0];
rz(-0.08188965344289881) q[0];
rx(-0.0047295138660964735) q[1];
rz(-0.0797217486975766) q[1];
rx(-0.14947644953812347) q[2];
rz(-0.009229019138552619) q[2];
rx(-0.12151889520981034) q[3];
rz(-0.06151819293509837) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06794358024571337) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.00401749310615462) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.051198517875179816) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11009467700861687) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.051674173107818314) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04535440047431951) q[3];
cx q[2],q[3];
rx(-0.11303134755365898) q[0];
rz(-0.0020422314790657097) q[0];
rx(0.04154631171279606) q[1];
rz(-0.025314802151033808) q[1];
rx(-0.12906987628258781) q[2];
rz(-0.03697274896570604) q[2];
rx(-0.17357105420331367) q[3];
rz(-0.02870898125721814) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12783717710787745) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.01605834500826996) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04079998955612018) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10803209423640465) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04366758367797569) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.025363806362716262) q[3];
cx q[2],q[3];
rx(-0.05472646565422753) q[0];
rz(0.01250607921136331) q[0];
rx(-0.0451841017242751) q[1];
rz(-0.06994826706796238) q[1];
rx(-0.060592615527054014) q[2];
rz(-0.08931002820490722) q[2];
rx(-0.15929078269768018) q[3];
rz(-0.01033126648067722) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08863602779435098) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0065680492883003275) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05597936245737523) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11546182841106327) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0040988158722471794) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03415184485848388) q[3];
cx q[2],q[3];
rx(-0.05177665401969978) q[0];
rz(0.04320107667379737) q[0];
rx(0.005296370928307627) q[1];
rz(-0.10018658845053267) q[1];
rx(-0.03300106214716894) q[2];
rz(-0.1061774786166576) q[2];
rx(-0.137400822362462) q[3];
rz(-0.055392416739786655) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10481766131034702) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.05142855846470173) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.007761487877824803) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09820858781101711) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.028265735714984967) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.017918800589056337) q[3];
cx q[2],q[3];
rx(-0.13751674966286062) q[0];
rz(0.046363010876129554) q[0];
rx(-0.014737525459915342) q[1];
rz(-0.06367697716644505) q[1];
rx(-0.04192693992849181) q[2];
rz(-0.05725687096536739) q[2];
rx(-0.13706185257869533) q[3];
rz(-0.008625758676382921) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17305153336886192) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0003809467628746924) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.004703630696440615) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18368548452015712) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07980373961620979) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07569079640380319) q[3];
cx q[2],q[3];
rx(-0.13641180257541347) q[0];
rz(0.016439687559030566) q[0];
rx(0.007238609026278801) q[1];
rz(-0.08320739049121922) q[1];
rx(-0.039435613841167484) q[2];
rz(-0.04702694732154564) q[2];
rx(-0.17709242797096925) q[3];
rz(-0.03123075148867863) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13059917747848804) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.018878908216224906) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05914707762787055) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.21363239742027532) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0019941805569029746) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.017173819437045813) q[3];
cx q[2],q[3];
rx(-0.1391792620600193) q[0];
rz(0.050988756594476894) q[0];
rx(0.017847910014731867) q[1];
rz(-0.08392951000111326) q[1];
rx(-0.05507889717679178) q[2];
rz(-0.07709395911729398) q[2];
rx(-0.1421403596614236) q[3];
rz(-0.0700000164394051) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1628915639627271) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.017626923573580103) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.11080839094082201) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18834312220924807) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.009394865343249105) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0071820165680632235) q[3];
cx q[2],q[3];
rx(-0.10854128580603292) q[0];
rz(-0.018755275028167356) q[0];
rx(-0.027213332977071014) q[1];
rz(-0.09920731971530045) q[1];
rx(-0.033416513837284986) q[2];
rz(-0.07137287163327445) q[2];
rx(-0.16911411785763883) q[3];
rz(-0.05374072764139833) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05141829972324729) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05490626213103658) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03785909679528296) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15757099199126098) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.009652933250510825) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0010648851537668738) q[3];
cx q[2],q[3];
rx(-0.09305683444810584) q[0];
rz(-0.025520461763771898) q[0];
rx(-0.01683876478824143) q[1];
rz(-0.0717433425319968) q[1];
rx(0.032825756347946856) q[2];
rz(-0.039466538403033664) q[2];
rx(-0.12835363240671138) q[3];
rz(-0.031032868954558708) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10765248302390223) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0014861307845152184) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03404017512750999) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14520597544018757) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.03497892553683069) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05171004026291057) q[3];
cx q[2],q[3];
rx(-0.12911129777242963) q[0];
rz(-0.01834539291043644) q[0];
rx(-0.01402166984068872) q[1];
rz(-0.03686992595693256) q[1];
rx(0.01419187514225629) q[2];
rz(-0.07508692142741928) q[2];
rx(-0.12310269633431821) q[3];
rz(0.01408496840563216) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10190128799808526) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0025244021356867403) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03151677614644932) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11676676457021323) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.014035884611081347) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.026732768876426234) q[3];
cx q[2],q[3];
rx(-0.1171333138573796) q[0];
rz(-0.026210160593750646) q[0];
rx(0.01266319803409588) q[1];
rz(-0.10406308977063923) q[1];
rx(0.032266216518316466) q[2];
rz(-0.05256880260574002) q[2];
rx(-0.18930821367178652) q[3];
rz(-0.011763068330356435) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09070940011478912) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.029065774596375146) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.049873680485717165) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07298020412260371) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0649643089345573) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05137248598467562) q[3];
cx q[2],q[3];
rx(-0.16876692855024616) q[0];
rz(-0.06099271539893372) q[0];
rx(-0.014905257923376347) q[1];
rz(-0.1285392311818039) q[1];
rx(-0.025259688247872874) q[2];
rz(-0.060777888257842506) q[2];
rx(-0.15698977418634194) q[3];
rz(0.02958728746625746) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0556952967227182) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.012663136725022292) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.011185660054946586) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.043361309065903963) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.06698793752255666) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.01151038228743599) q[3];
cx q[2],q[3];
rx(-0.18192161404890497) q[0];
rz(-0.048004036617977315) q[0];
rx(-0.029433616860458957) q[1];
rz(-0.11926758148293636) q[1];
rx(0.013677021562551463) q[2];
rz(-0.14439008027671282) q[2];
rx(-0.10820405563272419) q[3];
rz(0.02143438232895636) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.053366296633613655) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02126550207816954) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.02279624070972067) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08932747744248375) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.008330512629815061) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.006690474507154018) q[3];
cx q[2],q[3];
rx(-0.19375803451593934) q[0];
rz(-0.07344704113207147) q[0];
rx(0.0029555662171280787) q[1];
rz(-0.08440546259045123) q[1];
rx(0.013700431844580557) q[2];
rz(-0.16135430318046778) q[2];
rx(-0.14605262523051357) q[3];
rz(0.059804759981490166) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07068388147839594) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.036182044078276576) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.034048557021352635) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0998347278809576) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0030647020749021486) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.025235612198723393) q[3];
cx q[2],q[3];
rx(-0.20937181374766892) q[0];
rz(-0.0792797977381128) q[0];
rx(0.040357632662762925) q[1];
rz(-0.14831969886983348) q[1];
rx(-0.014656588662877706) q[2];
rz(-0.12201585465758626) q[2];
rx(-0.12139474680304713) q[3];
rz(0.06456047190609615) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1102274794271737) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.005529277896964) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.018836549498834596) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08603793584844335) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06723429521816632) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.043738901640875606) q[3];
cx q[2],q[3];
rx(-0.17191841986052836) q[0];
rz(-0.002521910620643535) q[0];
rx(-0.002039934448893931) q[1];
rz(-0.11933131635382811) q[1];
rx(0.06337075764981229) q[2];
rz(-0.11568069005365829) q[2];
rx(-0.14714680344257744) q[3];
rz(0.015744910078344537) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.030578686784644427) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02760433523165176) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04492720071539289) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12104501593276384) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07675185063744953) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05542279578507376) q[3];
cx q[2],q[3];
rx(-0.17588782672656295) q[0];
rz(-0.025861491051800793) q[0];
rx(0.006279499471309326) q[1];
rz(-0.15681988164551136) q[1];
rx(-0.03729036528867107) q[2];
rz(-0.157763111458304) q[2];
rx(-0.1849055903868776) q[3];
rz(-0.019030510830204326) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09676270226132343) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.008503789489026917) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.012475906027962901) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15961093040871221) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11598837293549433) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.012807200420015806) q[3];
cx q[2],q[3];
rx(-0.18940654625446307) q[0];
rz(-0.0027877831053278765) q[0];
rx(0.0017404000457616367) q[1];
rz(-0.11399779987951564) q[1];
rx(0.029498185398193846) q[2];
rz(-0.10734633351196474) q[2];
rx(-0.12743074416847255) q[3];
rz(-0.05134535131880512) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12171199722495098) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04677784456608826) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04489210376444916) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09174280191298079) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.058913394479753405) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04255185060107976) q[3];
cx q[2],q[3];
rx(-0.1953856166046124) q[0];
rz(-0.0452738474948124) q[0];
rx(0.02804279038260641) q[1];
rz(-0.11842019817860235) q[1];
rx(0.003099266923836853) q[2];
rz(-0.14331087622613542) q[2];
rx(-0.18519450308034657) q[3];
rz(-0.08632646640687577) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11920955769146595) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03710623023938018) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06465762903885634) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10493525313225055) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.040623208069878215) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.024944096360257195) q[3];
cx q[2],q[3];
rx(-0.20366820571401129) q[0];
rz(-0.0008752943455513824) q[0];
rx(0.003061870552912988) q[1];
rz(-0.04598717780663907) q[1];
rx(-0.015181174692259606) q[2];
rz(-0.11909716807648904) q[2];
rx(-0.1859928719920768) q[3];
rz(-0.025311774603227772) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12637849853189828) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.009603907632129387) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0011211751222357806) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09473158607261083) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0887109474117157) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03357156375636731) q[3];
cx q[2],q[3];
rx(-0.2805027731211128) q[0];
rz(-0.017119820226877972) q[0];
rx(-0.0448626382136653) q[1];
rz(-0.04521511909352696) q[1];
rx(-0.0952335374086907) q[2];
rz(-0.1356108587367183) q[2];
rx(-0.15381663038985263) q[3];
rz(-0.1143442507349812) q[3];