OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.2854819963299704) q[0];
rz(-2.837922557316731) q[0];
ry(2.4582373468730143) q[1];
rz(-1.3833868841178802) q[1];
ry(-0.4787555220666481) q[2];
rz(-1.5523242663882562) q[2];
ry(-3.0951669803093425) q[3];
rz(2.619867188899752) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.4664221148671315) q[0];
rz(1.0603240359089028) q[0];
ry(0.830929020209151) q[1];
rz(1.3928042881104739) q[1];
ry(-0.7569243423361547) q[2];
rz(1.5103990963666627) q[2];
ry(2.408262273847408) q[3];
rz(-2.438279339361148) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7811950178377494) q[0];
rz(2.326610433593459) q[0];
ry(2.0220617210939267) q[1];
rz(3.000430235468031) q[1];
ry(-1.9315482322692574) q[2];
rz(-0.038442203762879754) q[2];
ry(0.9112495617036451) q[3];
rz(-2.2037972935539925) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.4954620916545191) q[0];
rz(-2.6312592478517605) q[0];
ry(-2.2974888988930458) q[1];
rz(-2.4294164286382713) q[1];
ry(-0.4729174443890604) q[2];
rz(-1.8325884100792598) q[2];
ry(-1.3121016377292118) q[3];
rz(-2.822194184493438) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.29954884915613267) q[0];
rz(1.8594034939067823) q[0];
ry(-1.6579058854673896) q[1];
rz(-0.5421715211964031) q[1];
ry(0.38046431224192695) q[2];
rz(0.6822882581163833) q[2];
ry(0.6614189145936109) q[3];
rz(-1.3357903955796038) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.242590402649367) q[0];
rz(2.05062440538611) q[0];
ry(-1.6544948746285895) q[1];
rz(-2.780997349516336) q[1];
ry(-1.080084051262131) q[2];
rz(1.4871827907640431) q[2];
ry(-0.5850577573597135) q[3];
rz(0.6249488671372614) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.8083775314952079) q[0];
rz(0.5216869192820754) q[0];
ry(2.829356147167158) q[1];
rz(2.132666387668068) q[1];
ry(-2.4524638810710884) q[2];
rz(1.4372137461868717) q[2];
ry(1.0233057638998844) q[3];
rz(-0.1380461753315423) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.7729842860644935) q[0];
rz(-0.41985133078574366) q[0];
ry(2.769973011249043) q[1];
rz(2.179845651775304) q[1];
ry(1.9843958423913586) q[2];
rz(1.858378428478457) q[2];
ry(-2.0445025877620875) q[3];
rz(-3.0424940605360873) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.1308960866446047) q[0];
rz(0.7009467004957303) q[0];
ry(-2.7660425223304146) q[1];
rz(-1.5372033863769143) q[1];
ry(-0.2637742091434534) q[2];
rz(-0.02113713751112653) q[2];
ry(1.7184842537737062) q[3];
rz(-0.6159098110587511) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5289550224303161) q[0];
rz(-2.5480965539803133) q[0];
ry(-2.7376598820571725) q[1];
rz(-2.3617912129552847) q[1];
ry(2.288736064082584) q[2];
rz(-0.800869494386515) q[2];
ry(-2.930903997498105) q[3];
rz(-0.4106399793372928) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4140250952241713) q[0];
rz(-0.38421583736815973) q[0];
ry(1.6942905445252574) q[1];
rz(-2.9764590257800103) q[1];
ry(2.900809826982124) q[2];
rz(0.1466177101896031) q[2];
ry(-0.455031790108289) q[3];
rz(0.10534423696028253) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7579984907252086) q[0];
rz(2.189191761298557) q[0];
ry(1.6388582590872167) q[1];
rz(-0.1309350480660021) q[1];
ry(2.153441324972056) q[2];
rz(-0.7764747813748666) q[2];
ry(0.5599879350645605) q[3];
rz(-0.8073384894046964) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5958573652306427) q[0];
rz(-1.1331875715899282) q[0];
ry(2.2078388960101645) q[1];
rz(2.68368761061833) q[1];
ry(-3.0055539935294044) q[2];
rz(-1.2393420957153758) q[2];
ry(0.9280965472859801) q[3];
rz(2.0647937486104238) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.408523695270831) q[0];
rz(-2.523828441903904) q[0];
ry(2.8574921609341875) q[1];
rz(-2.643570645025495) q[1];
ry(-2.464592034001303) q[2];
rz(-1.6012196952868207) q[2];
ry(-1.2314232700251155) q[3];
rz(0.9862606657524228) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.9786154682493855) q[0];
rz(0.891717143126116) q[0];
ry(2.544170665590861) q[1];
rz(-0.9584823693335193) q[1];
ry(-0.3626738669070173) q[2];
rz(2.5990513800876522) q[2];
ry(-2.2674434010767532) q[3];
rz(-1.3525712789731017) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.5056913646635226) q[0];
rz(0.4779213468129449) q[0];
ry(1.8984635419695746) q[1];
rz(-0.7510743158030975) q[1];
ry(2.4843566125483267) q[2];
rz(0.2154107255697762) q[2];
ry(-1.3031783141834892) q[3];
rz(-0.9704929241133072) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.766081944939363) q[0];
rz(-1.0052576649940574) q[0];
ry(-1.0801819654126221) q[1];
rz(-0.29369836161434676) q[1];
ry(-1.6023079902222124) q[2];
rz(1.2083050482761946) q[2];
ry(-0.6033608049927381) q[3];
rz(0.036897290707767816) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.7640349731317428) q[0];
rz(1.638333403739721) q[0];
ry(-2.4723588747004865) q[1];
rz(-2.1219680721694756) q[1];
ry(0.49536833576166917) q[2];
rz(-1.972682643529333) q[2];
ry(2.824361684458908) q[3];
rz(3.0331140677375092) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.004668493745638713) q[0];
rz(-2.398476067306592) q[0];
ry(1.0886535159687796) q[1];
rz(0.5518967645804365) q[1];
ry(2.170905844955648) q[2];
rz(2.8428688366339645) q[2];
ry(0.24133880101436517) q[3];
rz(-2.3241243045470594) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.8749924589607927) q[0];
rz(2.4614779432738096) q[0];
ry(0.37176115341532157) q[1];
rz(0.3664192285929486) q[1];
ry(-1.9056476963224716) q[2];
rz(-2.282443229519326) q[2];
ry(0.3460718590061608) q[3];
rz(-0.21000784452942156) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.507891170244526) q[0];
rz(-2.610283748696612) q[0];
ry(1.7711245925957417) q[1];
rz(2.4531765205998197) q[1];
ry(-2.1325682629083804) q[2];
rz(-2.5107628838264797) q[2];
ry(-1.8405360916530888) q[3];
rz(2.768913688053772) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.07887047089480602) q[0];
rz(2.4255721203956733) q[0];
ry(1.4234638492108147) q[1];
rz(-1.6397082563547498) q[1];
ry(-2.863319526674156) q[2];
rz(1.3980811579261436) q[2];
ry(0.7178934615957678) q[3];
rz(-2.540185508671859) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6364036908933626) q[0];
rz(-2.718634388208316) q[0];
ry(-0.749367687624429) q[1];
rz(-2.9546469022744413) q[1];
ry(1.0696496719964879) q[2];
rz(-1.6084136002744585) q[2];
ry(1.5989041172431544) q[3];
rz(0.3122327491359584) q[3];