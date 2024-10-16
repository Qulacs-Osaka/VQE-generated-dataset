OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.570884666798249) q[0];
rz(-1.5707586354184475) q[0];
ry(1.5710351889531884) q[1];
rz(1.570808039466222) q[1];
ry(-1.5708122225500054) q[2];
rz(1.570810856799551) q[2];
ry(6.023431929763738e-05) q[3];
rz(-0.11300591893966595) q[3];
ry(-2.128789776281893) q[4];
rz(-1.5710258419484036) q[4];
ry(1.5707588894621587) q[5];
rz(-0.26686134617350454) q[5];
ry(-3.140777638142804) q[6];
rz(-1.824099343163752) q[6];
ry(1.5708331003594271) q[7];
rz(-0.29683532269585067) q[7];
ry(1.5711876936184446) q[8];
rz(3.141161462728899) q[8];
ry(-0.0071297294245823645) q[9];
rz(3.0647810661274155) q[9];
ry(1.5708055540935977) q[10];
rz(5.15519815502724e-05) q[10];
ry(-1.5708435152371314) q[11];
rz(3.141517235225946) q[11];
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
ry(-1.5706253949812001) q[0];
rz(-3.138982565447007) q[0];
ry(-1.5707875205117627) q[1];
rz(0.9077556761858155) q[1];
ry(-1.5707930532722498) q[2];
rz(2.876753251800678) q[2];
ry(-3.1415557486380483) q[3];
rz(-2.4048798261637927) q[3];
ry(1.5699170752227578) q[4];
rz(-1.624088324877686) q[4];
ry(3.141499764736223) q[5];
rz(-1.597230106476564) q[5];
ry(-1.5709576446542686) q[6];
rz(2.400918074434943e-06) q[6];
ry(3.141352778318924) q[7];
rz(1.2739658370425362) q[7];
ry(-2.4976309576661024) q[8];
rz(1.5704348741550644) q[8];
ry(-3.375377681215674e-05) q[9];
rz(1.009494058139908) q[9];
ry(0.6766176814993505) q[10];
rz(-1.5708063959966954) q[10];
ry(-1.1366629142770748) q[11];
rz(-0.552971108752339) q[11];
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
ry(-1.5747828202541196) q[0];
rz(1.6315283264194165) q[0];
ry(1.570768615197424) q[1];
rz(-2.925942050169884) q[1];
ry(3.141574874247329) q[2];
rz(1.0005711145061245) q[2];
ry(-1.5707191822581759) q[3];
rz(-1.570814135773918) q[3];
ry(-3.1415712197901264) q[4];
rz(2.8300759406951754) q[4];
ry(3.0405861501882607) q[5];
rz(1.5067619134528751) q[5];
ry(1.5519534737952467) q[6];
rz(-1.702932131534764) q[6];
ry(-2.299825749735532) q[7];
rz(-1.3797755675239562) q[7];
ry(-2.2020518340766104) q[8];
rz(1.5758757715295937) q[8];
ry(5.893408209570798e-07) q[9];
rz(0.3089829329683669) q[9];
ry(1.5326712164597975) q[10];
rz(0.7468643171107879) q[10];
ry(0.0731246222893267) q[11];
rz(0.554202678841414) q[11];
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
ry(3.1304228453396825) q[0];
rz(-0.9742599101513179) q[0];
ry(0.006057299364508495) q[1];
rz(2.2657187679878596) q[1];
ry(-1.5708005739023028) q[2];
rz(0.30942156369743495) q[2];
ry(1.5705373447903281) q[3];
rz(-1.7720232607367306) q[3];
ry(0.0002030784320430357) q[4];
rz(-1.3604830460590112) q[4];
ry(-0.0008073475053169687) q[5];
rz(-1.2674980931740376) q[5];
ry(3.141232511063547) q[6];
rz(-1.7065470270329017) q[6];
ry(5.1281591167475e-05) q[7];
rz(-1.405602535930786) q[7];
ry(-3.140145492820017) q[8];
rz(1.5771324097077626) q[8];
ry(-3.1415900179712057) q[9];
rz(-0.6860162168973226) q[9];
ry(0.0005233937114630307) q[10];
rz(2.3947099922663146) q[10];
ry(1.569715610457032) q[11];
rz(1.373680004241295) q[11];
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
ry(-1.5708490575436524) q[0];
rz(-1.3537081936286564) q[0];
ry(3.1415757170883194) q[1];
rz(-1.7087989903323795) q[1];
ry(-6.369762746771812e-05) q[2];
rz(-0.3094012604095857) q[2];
ry(3.141392385189002) q[3];
rz(-0.20117230483862247) q[3];
ry(-3.137323408408195) q[4];
rz(-0.1901848841258705) q[4];
ry(-1.5691371510598697) q[5];
rz(-1.1964668054529382) q[5];
ry(1.5730478577817155) q[6];
rz(-1.3048157243427312) q[6];
ry(-0.18114727684498622) q[7];
rz(-1.921555776281335) q[7];
ry(-3.068392296706033) q[8];
rz(2.84867283698977) q[8];
ry(3.141586796075088) q[9];
rz(2.6498360185159835) q[9];
ry(-1.812015226797087) q[10];
rz(-2.349289042615167) q[10];
ry(3.1410919664389207) q[11];
rz(-1.7679208658913668) q[11];
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
ry(-0.00035807594151648914) q[0];
rz(-0.21706001141947875) q[0];
ry(3.793969916809914e-05) q[1];
rz(1.3568270000635367) q[1];
ry(-1.570864268519508) q[2];
rz(-1.9328545788370528) q[2];
ry(-1.5707816272801407) q[3];
rz(0.8052520937700268) q[3];
ry(-3.1323286090448774) q[4];
rz(-1.831290944839415) q[4];
ry(0.00048109962521714067) q[5];
rz(2.767902458462125) q[5];
ry(3.130862642153531) q[6];
rz(1.3568030565186928) q[6];
ry(1.5703539819544698) q[7];
rz(-3.141530452021845) q[7];
ry(3.141560231783719) q[8];
rz(2.8469291825185223) q[8];
ry(2.074763491528176e-05) q[9];
rz(-1.4032472865157013) q[9];
ry(-5.814653615487941e-06) q[10];
rz(-0.7925475451391265) q[10];
ry(1.570691417634329) q[11];
rz(-1.5706256589754666) q[11];
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
ry(1.570785327452894) q[0];
rz(-1.2348970857446695) q[0];
ry(3.079320741848873) q[1];
rz(-0.195937154641407) q[1];
ry(-3.141133097587784) q[2];
rz(1.7275930553308374) q[2];
ry(-3.1415649196640203) q[3];
rz(-0.7650046462815975) q[3];
ry(-1.496811921104922) q[4];
rz(-3.0698787646456935) q[4];
ry(1.5422401922421214) q[5];
rz(2.0724207857564627) q[5];
ry(-1.5718495350507329) q[6];
rz(1.5732702019518516) q[6];
ry(-1.571185809884807) q[7];
rz(-2.7840364947813145) q[7];
ry(2.7609857734572305) q[8];
rz(2.3077651583993393) q[8];
ry(3.1344551982961266) q[9];
rz(-0.6928677777930559) q[9];
ry(1.960095422223572) q[10];
rz(1.5705805271056643) q[10];
ry(-1.5663117102640134) q[11];
rz(1.7830354841803606) q[11];
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
ry(0.00012196706396760959) q[0];
rz(-2.644156737179442) q[0];
ry(3.141587413142456) q[1];
rz(-0.4682411206359098) q[1];
ry(3.141591089601152) q[2];
rz(-0.2092408197499717) q[2];
ry(-3.0783864029926304) q[3];
rz(-1.3570814210992355) q[3];
ry(1.5747323608389692) q[4];
rz(-1.3222661464740302) q[4];
ry(3.137619774857738) q[5];
rz(-2.6242036630280436) q[5];
ry(-1.5746847755453643) q[6];
rz(-0.009503400902295346) q[6];
ry(0.0002140685532170394) q[7];
rz(1.4394498960420954) q[7];
ry(3.141578389685352) q[8];
rz(-0.8332924611936763) q[8];
ry(-3.1406632046351866) q[9];
rz(-2.8481838944617848) q[9];
ry(1.6152220466892713) q[10];
rz(1.570798477484228) q[10];
ry(3.1253168742693136) q[11];
rz(0.8525559941506478) q[11];
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
ry(-6.444068440636386e-06) q[0];
rz(0.08513333524482203) q[0];
ry(0.004688670959924757) q[1];
rz(2.513870391649356) q[1];
ry(3.1415825414637175) q[2];
rz(-3.1094565516786066) q[2];
ry(-0.00010139056441614527) q[3];
rz(1.3575205916916733) q[3];
ry(-0.016628046385372845) q[4];
rz(-0.23907092187608375) q[4];
ry(-1.5742256962260113) q[5];
rz(0.7082266047966953) q[5];
ry(-1.573826238272982) q[6];
rz(-2.7525519290292357) q[6];
ry(-0.00019933982807984236) q[7];
rz(-0.3329472406766098) q[7];
ry(-1.635641816889378) q[8];
rz(1.5704560779710945) q[8];
ry(-3.140617393020493) q[9];
rz(-2.1697935625331937) q[9];
ry(1.5061473204795481) q[10];
rz(-1.570969056366733) q[10];
ry(-1.8175632605325953e-05) q[11];
rz(-0.5173806800630928) q[11];
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
ry(0.003350682311926743) q[0];
rz(0.6586567840380138) q[0];
ry(-3.1415921629923917) q[1];
rz(-1.9516076852949373) q[1];
ry(3.1415904483540396) q[2];
rz(0.18114588120164576) q[2];
ry(-2.141122844762217) q[3];
rz(0.5338622304127547) q[3];
ry(-2.1793340411470155) q[4];
rz(3.141557458437547) q[4];
ry(0.09771578950114876) q[5];
rz(-1.664727097038769) q[5];
ry(-3.101880502059934e-06) q[6];
rz(1.16549109516614) q[6];
ry(-1.5707747640050516) q[7];
rz(1.882631702715957) q[7];
ry(1.6828994459238078) q[8];
rz(-3.141043782303029) q[8];
ry(7.200055629663636e-05) q[9];
rz(-3.020258593858255) q[9];
ry(0.9564699871348471) q[10];
rz(1.674185327438238e-05) q[10];
ry(1.4189741111118233) q[11];
rz(1.7375187925940523) q[11];
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
ry(3.14138953843331) q[0];
rz(3.0306217656059355) q[0];
ry(-3.1415452349729716) q[1];
rz(-2.9823821032974878) q[1];
ry(6.944653644147535e-06) q[2];
rz(-2.9793141049941783) q[2];
ry(-0.0011416771654300106) q[3];
rz(-1.0819742177401468) q[3];
ry(-1.5707814134130604) q[4];
rz(-2.498950913518098) q[4];
ry(-1.5707597693323894) q[5];
rz(-3.002597003573536) q[5];
ry(-1.5707747556546852) q[6];
rz(-3.015242536842655) q[6];
ry(-3.1414681821191217) q[7];
rz(-0.004565628312675151) q[7];
ry(-1.569153602854139) q[8];
rz(-2.572925709409496) q[8];
ry(1.5707989736719379) q[9];
rz(-3.143862059573621e-07) q[9];
ry(-1.5707969684755023) q[10];
rz(-3.0221376458222657) q[10];
ry(-1.5688270402774709) q[11];
rz(-0.02474485721144721) q[11];
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
ry(2.364617258857703e-05) q[0];
rz(0.24144127582916106) q[0];
ry(3.1415911691827034) q[1];
rz(-3.0299666201047124) q[1];
ry(-3.141583402856179) q[2];
rz(1.2780914079128876) q[2];
ry(-1.7279887457810617e-05) q[3];
rz(-0.86009518519396) q[3];
ry(3.141590151211996) q[4];
rz(0.7750041875089765) q[4];
ry(-9.283446127383854e-06) q[5];
rz(0.023927340139621833) q[5];
ry(-7.166257491132917e-08) q[6];
rz(1.5767832494052259) q[6];
ry(3.141574602238025) q[7];
rz(2.987849524912151) q[7];
ry(-3.1415923962860486) q[8];
rz(-2.1940341033354107) q[8];
ry(1.5707783250753862) q[9];
rz(-1.4084807951245466) q[9];
ry(3.141589578144004) q[10];
rz(2.0709637521305204) q[10];
ry(-3.141378865591617) q[11];
rz(0.13818692910849878) q[11];