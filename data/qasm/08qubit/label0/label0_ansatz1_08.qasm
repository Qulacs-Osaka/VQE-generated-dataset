OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.0785788618372045) q[0];
rz(-0.6716011728702916) q[0];
ry(-0.269829538642238) q[1];
rz(0.30309872415274874) q[1];
ry(2.9322332131186113) q[2];
rz(0.2191443096008097) q[2];
ry(-2.6354312690836017) q[3];
rz(-2.2608286563159092) q[3];
ry(2.696549739148778) q[4];
rz(-2.91313966728327) q[4];
ry(-3.064299929247852) q[5];
rz(0.5941549032269695) q[5];
ry(2.04171662627678) q[6];
rz(-1.5081121116433884) q[6];
ry(0.30316005034768084) q[7];
rz(-2.188725522733141) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.088069589357474) q[0];
rz(0.5839415813113279) q[0];
ry(3.121337770569957) q[1];
rz(0.7245951764160603) q[1];
ry(0.05888477354877737) q[2];
rz(0.8739504859164641) q[2];
ry(1.259333828604751) q[3];
rz(-0.6261622278773147) q[3];
ry(-0.5958160098554891) q[4];
rz(1.7905122337973136) q[4];
ry(-0.35065298854173754) q[5];
rz(3.090494890580596) q[5];
ry(0.3107530600653261) q[6];
rz(-1.7024785331240435) q[6];
ry(3.103322790490576) q[7];
rz(-0.5102191957500979) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.2810308860447384) q[0];
rz(-1.6534799938844313) q[0];
ry(-0.20607970828814975) q[1];
rz(-2.448971095330579) q[1];
ry(2.2960572722543624) q[2];
rz(2.758662631823281) q[2];
ry(-2.779902918178435) q[3];
rz(1.8880470478049638) q[3];
ry(0.005106579097727487) q[4];
rz(1.7284778140177712) q[4];
ry(-3.1263714901844915) q[5];
rz(1.0171879682891989) q[5];
ry(1.15816832380012) q[6];
rz(1.1207236956261588) q[6];
ry(-0.1417155593577622) q[7];
rz(0.9347404545049063) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.1045450610041473) q[0];
rz(0.976344147678101) q[0];
ry(0.9195416219394349) q[1];
rz(-2.6931323624619696) q[1];
ry(1.3210678898242962) q[2];
rz(-0.06295780561546646) q[2];
ry(-1.2763202440492831) q[3];
rz(-1.9385190350014576) q[3];
ry(0.9468278168019879) q[4];
rz(-0.22587238212585256) q[4];
ry(-1.4776046272104584) q[5];
rz(-1.4623061540390596) q[5];
ry(-3.063397971935443) q[6];
rz(0.08710357923009136) q[6];
ry(-2.9196123191304584) q[7];
rz(2.8783492360631775) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.6117319124144599) q[0];
rz(-2.182850612218278) q[0];
ry(0.152865876173645) q[1];
rz(-2.6228451108411344) q[1];
ry(-0.09803103261422448) q[2];
rz(2.414692907306684) q[2];
ry(-3.0921162522419374) q[3];
rz(-2.1775187334502903) q[3];
ry(-0.0031273589079084366) q[4];
rz(-3.078568564625019) q[4];
ry(2.0191139237027413) q[5];
rz(0.5125403083071989) q[5];
ry(1.219625925852291) q[6];
rz(-2.0860886298138572) q[6];
ry(0.5171018467499396) q[7];
rz(-0.3407382849417438) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.17203339818415841) q[0];
rz(0.7115120736417984) q[0];
ry(0.9697228780768334) q[1];
rz(1.9202463774939524) q[1];
ry(2.7100569246850315) q[2];
rz(-3.0206183476690613) q[2];
ry(-1.714380650913536) q[3];
rz(2.490271584342087) q[3];
ry(0.7517316077111664) q[4];
rz(-0.5641881132499686) q[4];
ry(3.1293392565433065) q[5];
rz(-0.291071515713706) q[5];
ry(2.698828531781478) q[6];
rz(1.2965530465600317) q[6];
ry(-2.9359190285701873) q[7];
rz(-1.4490608455233973) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.6472308217169391) q[0];
rz(-1.1701005727257159) q[0];
ry(2.0639580630613734) q[1];
rz(1.2382233455378495) q[1];
ry(1.5988754101791467) q[2];
rz(0.29234427536071733) q[2];
ry(-0.035195825208750975) q[3];
rz(-0.8038935719775698) q[3];
ry(-0.028627216420654907) q[4];
rz(0.09428864730889597) q[4];
ry(2.9949549233620236) q[5];
rz(3.0566574616594977) q[5];
ry(-1.1164772683799826) q[6];
rz(2.7287914709348766) q[6];
ry(0.018525762842374505) q[7];
rz(-1.50505680188034) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.0061475492441447) q[0];
rz(0.03518187721355659) q[0];
ry(0.3842568892086004) q[1];
rz(1.8030604922904105) q[1];
ry(-3.066712409266369) q[2];
rz(2.2469579402546254) q[2];
ry(-0.0644716809992491) q[3];
rz(2.9214919623494366) q[3];
ry(-0.774439915485587) q[4];
rz(-0.5493223413138076) q[4];
ry(0.1511972038101185) q[5];
rz(1.3416250145189637) q[5];
ry(-2.9065537087495956) q[6];
rz(2.7255523984546706) q[6];
ry(-2.577231576381601) q[7];
rz(-2.166094493985457) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.2035140164915366) q[0];
rz(-0.7582535556086497) q[0];
ry(1.7102053343881254) q[1];
rz(-1.4108955022630922) q[1];
ry(-1.6748107572341902) q[2];
rz(-1.4559989621014322) q[2];
ry(0.027449175038690043) q[3];
rz(-1.1427548427541685) q[3];
ry(-3.125815409767811) q[4];
rz(2.815971245965259) q[4];
ry(-2.4448491777471837) q[5];
rz(-0.11191106217260785) q[5];
ry(-2.760809070721939) q[6];
rz(1.8837628897971888) q[6];
ry(-2.832469149015388) q[7];
rz(-0.9464923738804298) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8929916715616129) q[0];
rz(1.3201445244407084) q[0];
ry(2.5296022101791658) q[1];
rz(1.4901225591513025) q[1];
ry(2.433256217206268) q[2];
rz(3.0744224100806736) q[2];
ry(0.03134031176587282) q[3];
rz(0.0342989292877386) q[3];
ry(-0.07557019477851323) q[4];
rz(-1.4527973363780324) q[4];
ry(0.19586525766192095) q[5];
rz(-1.894430784523572) q[5];
ry(-1.5283513242601823) q[6];
rz(-1.3743829569153618) q[6];
ry(-1.7462584450129395) q[7];
rz(-2.504713023493781) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.23156622490870687) q[0];
rz(1.9103880761122953) q[0];
ry(-0.005042897476008611) q[1];
rz(-1.3983185213101188) q[1];
ry(1.6516211545798887) q[2];
rz(0.9758709715447603) q[2];
ry(0.04476877123651683) q[3];
rz(2.448179888018471) q[3];
ry(-0.010800382764805327) q[4];
rz(-2.589733186960971) q[4];
ry(3.1170399288948087) q[5];
rz(0.8798451680308823) q[5];
ry(-3.1308266434391196) q[6];
rz(1.9986158056664045) q[6];
ry(-1.5669219789593447) q[7];
rz(-3.1147623321628246) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.3522240073478606) q[0];
rz(-2.4881414811509712) q[0];
ry(-2.9297557713772653) q[1];
rz(2.9160134679963168) q[1];
ry(2.1424718302317958) q[2];
rz(2.985663934021867) q[2];
ry(0.2970415991651896) q[3];
rz(-2.213426966335809) q[3];
ry(3.002534151333463) q[4];
rz(1.07848352725625) q[4];
ry(-0.1927323102859052) q[5];
rz(-0.7210012078466342) q[5];
ry(-0.2648592392924636) q[6];
rz(-1.8850736300500681) q[6];
ry(2.6319711012536713) q[7];
rz(-1.5343778877979022) q[7];