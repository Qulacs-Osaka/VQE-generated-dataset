OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.6235531233288396) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.5861404394787739) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09012457264526372) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.4292972601444551) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.3192530353898169) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.38495666700533593) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.6838212208887892) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.4848463767329814) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-2.9131240667337677) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.11217704615414434) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.12155349673888638) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-1.8473011442590048) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.8294618161925452) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.8643868586769392) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.6389150056695412) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-2.0249235173084097) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.772205370989922) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.92261023706694) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-1.2706766985831028) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.9045001092354211) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(1.7356078397101906) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.08311474250253233) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.0864554405577033) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.5342495264400572) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-1.6258235679936603) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(1.5508858617053858) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-2.8403087141125094) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.19731182093178484) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.2616511220705742) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-2.0693764608356204) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(1.5442104924292324) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(1.5112723644484627) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.07017701706480066) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(-1.2381020445891875) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.8575289224838882) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.6077551908459832) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-2.1359410831609513) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.356743685594135) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(1.7190719600143003) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(0.06444949100206512) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(-0.028051310183954733) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(1.839644166304042) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(-1.5896942836197323) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(-1.5698898440319273) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(1.598804521283103) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(-1.1348411074674674) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(-1.0513408735997167) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(1.4220357990444012) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(1.5504531043540881) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(1.5875955087352744) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(-1.5997962763380835) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(-0.049700669765651496) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(0.04714349592122115) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(0.3160717220134248) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(0.013438612957975386) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(0.013820503238725784) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(-0.014163321228354835) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(0.8416422432081703) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(0.7353270898966247) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(-2.842589567463952) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(-0.3637598969626028) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(-0.5440608554886216) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(0.2353297983813587) q[11];
cx q[9],q[11];
rx(-0.0018426589666997268) q[0];
rz(2.370195378114981) q[0];
rx(-0.0014348694839544296) q[1];
rz(-1.4864514512732956) q[1];
rx(-0.00023073032366965473) q[2];
rz(1.1627027054064822) q[2];
rx(-0.0005416398546087068) q[3];
rz(-0.5337594399149347) q[3];
rx(0.0016290000675621018) q[4];
rz(0.9675974823535702) q[4];
rx(-0.00010953676014675132) q[5];
rz(0.2545476425882172) q[5];
rx(-0.0007680157636424067) q[6];
rz(0.21807608989463734) q[6];
rx(0.013402976442236382) q[7];
rz(-0.3803581249253509) q[7];
rx(-0.0010670021626059953) q[8];
rz(-1.4045215415335965) q[8];
rx(-0.0013543066257972406) q[9];
rz(-0.2691915451624858) q[9];
rx(-5.001678305270634e-05) q[10];
rz(-2.968145954947455) q[10];
rx(0.002443861672646781) q[11];
rz(2.7013382725416184) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.13592785712953) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.4727297789443707) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(1.269553796962231) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.279327530071711) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.2921442596947235) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-1.178846942536256) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(3.1213424680886104) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04530200987890597) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.007005937748500496) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.18237091633431926) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(1.3093008007842382) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-1.5626483496638532) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.17751157198290438) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.3763156948519864) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-2.6863776581076233) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.5164578360319684) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(1.106119304588754) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(2.390132923868156) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(1.8706705960484158) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-1.2792625925751087) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(1.7788574022246808) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.1563195713409087) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.1419333903676343) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.1488589116465594) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.9701125129809435) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.9287048789524812) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(2.1511229176386655) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.7288249328527271) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.7345712993847612) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.7725862410467776) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.08281197129931779) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.0012176377729996935) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.07610878228379846) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(-1.80488604205218) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-1.1258110665300258) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-1.1145225168343154) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.0061770182613127035) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.03433271229387635) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.056481977717916416) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(-0.02174944272485444) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(-0.02842943574652661) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(-0.03698204494952048) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(-0.08299784389358941) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(-0.06720022064629083) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(0.09106440009662357) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(1.3985649205759252) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(-1.7400209392608206) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(-1.4498464457023823) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(0.01737235315632436) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(0.022218576060430473) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(-0.12914754691051822) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(0.039295753561276986) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(-0.04790173375864109) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(0.03359544305032859) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(0.07502066576743921) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(-0.07946785733061927) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(0.06824961342318614) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(-0.7339017990431876) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(-0.7780884333023516) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(0.5446530771042213) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(0.1831209824810523) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(0.20662457318575048) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(-0.3502838708385811) q[11];
cx q[9],q[11];
rx(-0.0006386681118088573) q[0];
rz(0.9026023729020132) q[0];
rx(-0.0018494275875883752) q[1];
rz(0.6375509232637384) q[1];
rx(0.013334373633047299) q[2];
rz(1.391873516690445) q[2];
rx(0.0015242778869622266) q[3];
rz(0.14544254138784563) q[3];
rx(0.00013035041253796304) q[4];
rz(-1.2936412497603698) q[4];
rx(0.00020197841807297046) q[5];
rz(-0.23388207567632505) q[5];
rx(0.00021765161754334385) q[6];
rz(-1.305591158090968) q[6];
rx(0.0010049353689198565) q[7];
rz(1.4379845992613984) q[7];
rx(0.0035228266731210213) q[8];
rz(-1.7274494283898856) q[8];
rx(-0.003101556415145139) q[9];
rz(-1.4265130928034626) q[9];
rx(-0.0030407271947675537) q[10];
rz(1.4348498406419743) q[10];
rx(0.004429617462001688) q[11];
rz(-1.377604182777749) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.7633515531054738) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.32121952106486745) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-1.1117650704359474) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.5321437661716146) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.947489262661219) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-1.5136247933738172) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.040260611944963796) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.037557284132035025) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.004978806539430648) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(1.3916619641654733) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.3455815779990365) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-1.6793462566789106) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.03624119654559438) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.09205701542883409) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.09651821209760297) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.0653848505448642) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.031043709465688813) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.003583689797656194) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.00022292351455631296) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.007526889777226548) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.011497695943673898) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.0374305128013757) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.03545709176537776) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.06577381947526782) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.40129368028630463) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.3346402511773153) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.2791955207878578) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.19750318779157502) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.19947736123382165) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.2829894058375771) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(1.0304746096677244) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-1.1135703428238506) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(1.0941052084116498) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.1596974026225716) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.173457253690304) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.659486674225559) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.5509669132132504) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.529524437795815) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(3.037919330071118) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(0.2623824668800113) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(-0.25777585787972934) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(-0.30090934004918674) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(0.05300430606545489) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(0.05619047376081841) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(-0.2181210310098817) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(-2.2359381382194545) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(0.7601772210483437) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(1.0542277928914716) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(2.6804610519960144) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(2.6082024330391365) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(-0.3667952253092628) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(-0.3587510767657944) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(-0.3416775309768837) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(-0.3332536776033063) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(-0.42703241986075813) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(0.4391804463120906) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(0.4317445171033633) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(-0.8803171727944815) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(-0.8956147322622018) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(-0.8972463281327275) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(0.7223943214257779) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(-2.390079335926429) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(-0.7193912799720606) q[11];
cx q[9],q[11];
rx(0.0005139134533138498) q[0];
rz(2.32956135569539) q[0];
rx(-0.0048042789719462764) q[1];
rz(0.09528758617836079) q[1];
rx(0.0007581946154849679) q[2];
rz(0.0336989324467371) q[2];
rx(-0.003818803461830754) q[3];
rz(0.09104633084220587) q[3];
rx(0.00102656798283563) q[4];
rz(0.025550202660171394) q[4];
rx(-0.0033685546817502666) q[5];
rz(-0.05652784781345885) q[5];
rx(-0.0005151900128191931) q[6];
rz(0.01645433124165785) q[6];
rx(-0.003482198432398161) q[7];
rz(-0.04357009518267453) q[7];
rx(-0.0005854477055327913) q[8];
rz(0.03006798828232943) q[8];
rx(0.0031021813801686577) q[9];
rz(3.107711089688867) q[9];
rx(-0.0008107318240944621) q[10];
rz(0.019918934248770098) q[10];
rx(-0.0041316470907144884) q[11];
rz(-0.040418034201421114) q[11];