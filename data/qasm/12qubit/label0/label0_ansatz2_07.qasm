OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.572839000025498) q[0];
rz(-1.0079968485222004) q[0];
ry(1.3928425086882514) q[1];
rz(0.0935931512255574) q[1];
ry(0.06144537658180909) q[2];
rz(3.1313972468165336) q[2];
ry(0.0030255318470887305) q[3];
rz(1.5158656483954132) q[3];
ry(0.0001376421958765306) q[4];
rz(-2.667196935528702) q[4];
ry(-0.15376523205475578) q[5];
rz(2.4488325723793842) q[5];
ry(-1.4585978289838142) q[6];
rz(-0.7538109738181579) q[6];
ry(-0.00530096777739697) q[7];
rz(-0.020523752441601317) q[7];
ry(-0.015427529059637735) q[8];
rz(0.15633505042388066) q[8];
ry(-1.5712104661109834) q[9];
rz(-0.0013305227946463036) q[9];
ry(1.4779863183776216) q[10];
rz(1.1439696457243969) q[10];
ry(-1.5700621693412764) q[11];
rz(-0.0002969796701366434) q[11];
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
ry(3.1415677845746277) q[0];
rz(-2.579266815076623) q[0];
ry(-3.141203521531711) q[1];
rz(-0.9143963284792735) q[1];
ry(-3.1412135937337355) q[2];
rz(-3.0255567436469293) q[2];
ry(0.0014299282047060984) q[3];
rz(2.5593271449156374) q[3];
ry(1.0825623005573897e-05) q[4];
rz(-2.911239875872653) q[4];
ry(0.06609032898685548) q[5];
rz(-1.5869061238308468) q[5];
ry(2.7553639207330702) q[6];
rz(2.460142130407986) q[6];
ry(1.5391086747912017) q[7];
rz(-3.108136334871771) q[7];
ry(2.8976749064978176) q[8];
rz(2.074662899931301) q[8];
ry(-0.06382714874992244) q[9];
rz(-1.56963877038025) q[9];
ry(3.138389465946115) q[10];
rz(1.210494195408825) q[10];
ry(-1.2813509479719871) q[11];
rz(1.5705963447582405) q[11];
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
ry(2.059573698342465) q[0];
rz(-1.5676192899167993) q[0];
ry(-3.1402200176398742) q[1];
rz(-2.5648974685067016) q[1];
ry(-3.0921414683965556) q[2];
rz(-1.506568145668144) q[2];
ry(-2.908657570548629) q[3];
rz(-2.3364885500886268) q[3];
ry(0.0006003925393929797) q[4];
rz(2.4078817098276164) q[4];
ry(-0.248866064859671) q[5];
rz(-2.9980168068164788) q[5];
ry(3.074431236999425) q[6];
rz(0.2690658610289471) q[6];
ry(-1.5363665541492262) q[7];
rz(0.6154684576754974) q[7];
ry(-0.013187832172370406) q[8];
rz(-0.5758090210245402) q[8];
ry(-2.0279800662963927) q[9];
rz(-3.1414248029913687) q[9];
ry(0.0004148862223267713) q[10];
rz(2.766445181263237) q[10];
ry(-2.4863942202322296) q[11];
rz(1.5713041130312895) q[11];
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
ry(2.6483906372616257) q[0];
rz(0.039947370380553515) q[0];
ry(1.5679626299069802) q[1];
rz(-1.5642434315144451) q[1];
ry(3.0566796237232263) q[2];
rz(-1.5806447504593901) q[2];
ry(0.02575147633618169) q[3];
rz(-0.8268623019815743) q[3];
ry(-1.5708113880846237) q[4];
rz(-0.003426053089183336) q[4];
ry(3.1236946193657262) q[5];
rz(0.077876596643374) q[5];
ry(0.006804115199156019) q[6];
rz(-1.728986141084393) q[6];
ry(3.1398366771797552) q[7];
rz(1.0643395863441114) q[7];
ry(-0.010358992459442686) q[8];
rz(-0.3038765814417826) q[8];
ry(-1.5709397284469764) q[9];
rz(-0.5394906595569396) q[9];
ry(-2.3633920346163704) q[10];
rz(-0.9271788319079439) q[10];
ry(2.0047912804896315) q[11];
rz(2.7602196303817297) q[11];
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
ry(0.24921636101385475) q[0];
rz(-2.472378784140749) q[0];
ry(-1.434784356705994) q[1];
rz(-1.5827948093984805) q[1];
ry(2.9873709031896434) q[2];
rz(1.5832315311819745) q[2];
ry(-0.08346981035676516) q[3];
rz(0.4294036803598562) q[3];
ry(3.1406040807271) q[4];
rz(-1.237089720217483) q[4];
ry(0.00042637077731952203) q[5];
rz(2.5421722812846954) q[5];
ry(-3.140399426289222) q[6];
rz(0.4818531296076296) q[6];
ry(-9.404530945555935e-05) q[7];
rz(1.1442312349903216) q[7];
ry(9.324152811007735e-05) q[8];
rz(-2.7831098022922838) q[8];
ry(-3.1414651257542787) q[9];
rz(0.7984731171985481) q[9];
ry(0.0004119722797951712) q[10];
rz(-2.190700913360949) q[10];
ry(3.141473230198352) q[11];
rz(-0.3818918502858298) q[11];
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
ry(3.1091299570936255) q[0];
rz(-0.320879695485635) q[0];
ry(-1.5674160339238372) q[1];
rz(3.1316273913900585) q[1];
ry(0.016829085442125438) q[2];
rz(3.091815043757661) q[2];
ry(-0.002087616772953634) q[3];
rz(-1.965132140906534) q[3];
ry(-3.140669266074895) q[4];
rz(1.9078534164719811) q[4];
ry(3.1403301211991215) q[5];
rz(1.0352457589061235) q[5];
ry(-0.0002710631701932087) q[6];
rz(1.7937628318153356) q[6];
ry(3.1346678052118104) q[7];
rz(-0.006886133360057834) q[7];
ry(0.056497004510887315) q[8];
rz(3.132186381783428) q[8];
ry(0.0006175990162953443) q[9];
rz(1.0836534647792733) q[9];
ry(3.1327689026851764) q[10];
rz(0.21698599761244064) q[10];
ry(-1.5523103611263895) q[11];
rz(1.574764932413108) q[11];
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
ry(-0.01026114143646369) q[0];
rz(-0.8360247276556789) q[0];
ry(-1.5828010991746329) q[1];
rz(3.1301118110762247) q[1];
ry(-2.259119010470681) q[2];
rz(-0.00011134410200952516) q[2];
ry(0.3035065870945894) q[3];
rz(-0.017636436758310843) q[3];
ry(-1.6508043724200883) q[4];
rz(3.039716383190876) q[4];
ry(-0.005914326283808613) q[5];
rz(2.414561135496428) q[5];
ry(-0.005699769342497163) q[6];
rz(2.9996144618788563) q[6];
ry(3.121468986042293) q[7];
rz(-0.6893429067146861) q[7];
ry(0.055390801972118646) q[8];
rz(0.1404519569377123) q[8];
ry(0.0005281559452775033) q[9];
rz(1.9694904989154156) q[9];
ry(-3.138108097548893) q[10];
rz(1.5148220435715496) q[10];
ry(2.8922645789079477) q[11];
rz(-3.110186039556086) q[11];
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
ry(0.014721504048680513) q[0];
rz(-0.9844332918762514) q[0];
ry(0.01312879946139092) q[1];
rz(2.2311775227814903) q[1];
ry(2.285689258436763) q[2];
rz(2.2007850403934963) q[2];
ry(2.9496505297273394) q[3];
rz(2.2971655300065383) q[3];
ry(3.141581911652412) q[4];
rz(1.7277028293513677) q[4];
ry(-3.1396975127225164) q[5];
rz(-2.2227122972535445) q[5];
ry(-3.138356505045792) q[6];
rz(-2.4998361862859144) q[6];
ry(0.0027654312520211523) q[7];
rz(0.6639176274843581) q[7];
ry(3.13520973385752) q[8];
rz(0.10918194064455466) q[8];
ry(2.8220434534097233e-05) q[9];
rz(-1.675597935361746) q[9];
ry(-0.43153031805533976) q[10];
rz(-3.111463994942591) q[10];
ry(-0.5950683215595634) q[11];
rz(-0.022997965915409813) q[11];
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
ry(-0.0016080690307243017) q[0];
rz(-1.4338708734261782) q[0];
ry(0.0010079699919440274) q[1];
rz(2.3289880254414577) q[1];
ry(3.139462922540285) q[2];
rz(2.4421923939318457) q[2];
ry(3.139278455260902) q[3];
rz(-3.139250346587732) q[3];
ry(-3.141142395757263) q[4];
rz(0.25883778648852646) q[4];
ry(0.010362207795909845) q[5];
rz(1.8344773690043665) q[5];
ry(3.1120183460621367) q[6];
rz(-0.08478447866295738) q[6];
ry(-3.0619035581603162) q[7];
rz(-0.19003910362842372) q[7];
ry(-2.920590250563912) q[8];
rz(1.8786008794621853) q[8];
ry(1.5702082861911597) q[9];
rz(3.1414510572273104) q[9];
ry(-1.60057539357048) q[10];
rz(-0.19451719431697087) q[10];
ry(1.4101119397388624) q[11];
rz(-1.915279133356953) q[11];
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
ry(-3.501340045630741e-05) q[0];
rz(-0.4262743110188314) q[0];
ry(3.1403813520216866) q[1];
rz(-1.6985526741723562) q[1];
ry(0.0001998863316335238) q[2];
rz(-1.8067157115294867) q[2];
ry(-3.141447862488824) q[3];
rz(-0.7263465599885812) q[3];
ry(-1.5706020447280669) q[4];
rz(-1.5639474464530698) q[4];
ry(3.141437463724728) q[5];
rz(-1.3521316890510722) q[5];
ry(3.1410033563763666) q[6];
rz(-2.6015822270867286) q[6];
ry(4.5513773470222694e-05) q[7];
rz(0.2030236498867894) q[7];
ry(-3.1415304314234014) q[8];
rz(-1.2690506800623318) q[8];
ry(-1.5712270382770208) q[9];
rz(3.139686743659385) q[9];
ry(3.1412910029212733) q[10];
rz(-0.27807920549269366) q[10];
ry(-3.1414949047942247) q[11];
rz(-1.7958112433425328) q[11];
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
ry(-0.03746832453266791) q[0];
rz(1.567423939667464) q[0];
ry(0.03943108404728334) q[1];
rz(1.5428746397346442) q[1];
ry(-1.607801161992934) q[2];
rz(-3.1404021115756313) q[2];
ry(1.5514008519644182) q[3];
rz(-3.125364355560055) q[3];
ry(0.5998777110757469) q[4];
rz(1.5585080311643547) q[4];
ry(-1.668667441517622) q[5];
rz(0.10358635604384148) q[5];
ry(-0.15260171557653737) q[6];
rz(-0.7738896526615042) q[6];
ry(-1.5695596956737294) q[7];
rz(-0.06985167754346072) q[7];
ry(1.5761668526877326) q[8];
rz(3.12884029192882) q[8];
ry(-1.5859844391670253) q[9];
rz(3.129345893643591) q[9];
ry(-1.3443787951647639) q[10];
rz(-1.1772407898812656) q[10];
ry(-1.3820283947140595) q[11];
rz(1.0144580516006192) q[11];