OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.6324305839601048) q[0];
rz(-2.570002149232342) q[0];
ry(1.5157903855106403) q[1];
rz(-0.7687377113618609) q[1];
ry(-3.0699478648021907) q[2];
rz(-0.12750372471422194) q[2];
ry(1.5524538927861822) q[3];
rz(-1.5577877189194957) q[3];
ry(-0.0007099944153736716) q[4];
rz(2.513722694761134) q[4];
ry(2.2164979698970937) q[5];
rz(0.02614661707692125) q[5];
ry(3.141069512395196) q[6];
rz(-2.72497605138533) q[6];
ry(-0.03997221373538906) q[7];
rz(-0.5748886642195057) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.0199769181667966) q[0];
rz(-1.8955619801026105) q[0];
ry(-2.2409785667199156) q[1];
rz(1.6220505076127933) q[1];
ry(1.5447518583704287) q[2];
rz(-2.527331310558492) q[2];
ry(2.1818159386358933) q[3];
rz(0.22602235340311563) q[3];
ry(-2.6970547984936952) q[4];
rz(-1.5719016923836975) q[4];
ry(-2.494850961050885) q[5];
rz(-1.638490746743043) q[5];
ry(1.5709752026122539) q[6];
rz(3.133251151877356) q[6];
ry(1.5372450022713569) q[7];
rz(-3.119856145283541) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-3.0643572554872094) q[0];
rz(1.905543197848469) q[0];
ry(2.1473021154772445) q[1];
rz(-1.544900500795496) q[1];
ry(-0.41054210707357086) q[2];
rz(2.7087866448784212) q[2];
ry(-0.010381483350236032) q[3];
rz(2.93842215128345) q[3];
ry(1.5704958181989637) q[4];
rz(1.5282911384666282) q[4];
ry(-1.3427834759029333) q[5];
rz(0.45095206109477903) q[5];
ry(-1.5708263238292846) q[6];
rz(-3.054865457013091) q[6];
ry(1.5708426757556628) q[7];
rz(-3.1305811534572294) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8384900560210484) q[0];
rz(0.04890947612781189) q[0];
ry(1.5995128232786548) q[1];
rz(-1.9270267592836845) q[1];
ry(-0.10277967563721457) q[2];
rz(-2.7475912434761103) q[2];
ry(-1.5328651003114508) q[3];
rz(-0.002623116292546404) q[3];
ry(1.5491264879473055) q[4];
rz(0.00015409934616528176) q[4];
ry(-1.5708018719257415) q[5];
rz(-2.5595945318678393) q[5];
ry(1.553348078648141) q[6];
rz(2.081168254155754) q[6];
ry(-1.5726761755263035) q[7];
rz(0.7574259769022725) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.5439941751955453) q[0];
rz(1.6339018009133608) q[0];
ry(3.1413860117392516) q[1];
rz(1.2136840484716374) q[1];
ry(2.585904071399948) q[2];
rz(-3.1412233582768256) q[2];
ry(2.6969351905928196) q[3];
rz(-3.1414304603337078) q[3];
ry(1.570791812735672) q[4];
rz(1.215396951330272) q[4];
ry(5.117362258921788e-05) q[5];
rz(-0.7763277037062534) q[5];
ry(1.397385715180424) q[6];
rz(2.77197950550834) q[6];
ry(0.001035302027019807) q[7];
rz(-1.6820306151850983) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.5948243422538273) q[0];
rz(2.8511576309259454) q[0];
ry(2.1875090687047383) q[1];
rz(-1.5442232543216479) q[1];
ry(-1.5699634323754932) q[2];
rz(-3.141137624356605) q[2];
ry(-1.570799995507331) q[3];
rz(0.5607981495544192) q[3];
ry(3.0170964451580944) q[4];
rz(2.158411871648232) q[4];
ry(3.140373115718976) q[5];
rz(-0.36010753942409574) q[5];
ry(-2.320663836058067) q[6];
rz(3.0048903167679994) q[6];
ry(-3.127314549148485) q[7];
rz(-0.08710129549687196) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.5707329311966338) q[0];
rz(0.0007924415241904187) q[0];
ry(0.16288839634567429) q[1];
rz(1.5497269659182535) q[1];
ry(1.5708006072363387) q[2];
rz(3.085957598734913) q[2];
ry(-3.141583487490568) q[3];
rz(-2.580782110057911) q[3];
ry(-3.4653179373376295e-05) q[4];
rz(2.198316544607563) q[4];
ry(3.1415780644208193) q[5];
rz(-2.4790571296276838) q[5];
ry(-2.9174742427386535) q[6];
rz(1.690357464349513) q[6];
ry(-5.0351823699824195e-05) q[7];
rz(-0.7489807795166148) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.5757908028281085) q[0];
rz(3.0238588832729585) q[0];
ry(-1.5707837516440915) q[1];
rz(-1.5707953288848113) q[1];
ry(-3.140753585328851) q[2];
rz(-1.6264315293870744) q[2];
ry(-0.34478139318786205) q[3];
rz(-1.5708206767456563) q[3];
ry(-1.5902422251365762) q[4];
rz(1.687483392784381) q[4];
ry(3.1412711561938464) q[5];
rz(-0.7427069733664701) q[5];
ry(1.8344670891324677) q[6];
rz(-2.2723042737633667) q[6];
ry(1.5802621458959507) q[7];
rz(-3.1352866630440954) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-3.5756475105230834e-06) q[0];
rz(-0.08731739321633863) q[0];
ry(-1.570795966361387) q[1];
rz(-0.22488718345788014) q[1];
ry(-1.57079759341043) q[2];
rz(-1.7819204542582598) q[2];
ry(-1.5707983483653425) q[3];
rz(2.9063968533700058) q[3];
ry(1.570768499952214) q[4];
rz(1.323742710298732) q[4];
ry(-1.5707433501787307) q[5];
rz(-1.8069257461177086) q[5];
ry(1.5710645631718452) q[6];
rz(-1.776429304199631) q[6];
ry(1.155017313175802e-05) q[7];
rz(-1.8128176819031916) q[7];