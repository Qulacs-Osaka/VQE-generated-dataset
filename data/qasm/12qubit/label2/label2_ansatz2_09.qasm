OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5706932404653136) q[0];
rz(3.427832980129608e-06) q[0];
ry(1.3094573847928457) q[1];
rz(-1.8616564553391743) q[1];
ry(1.5707929811268102) q[2];
rz(3.141581113676525) q[2];
ry(-1.525551821081514) q[3];
rz(-0.060754424639729784) q[3];
ry(-1.5711083567619384) q[4];
rz(-0.3307124121473279) q[4];
ry(-1.5195054100091756) q[5];
rz(-2.0595240453567305) q[5];
ry(2.550147080524929) q[6];
rz(-1.571248235620275) q[6];
ry(0.0022716324909534276) q[7];
rz(-1.8673224339476289) q[7];
ry(1.570652531409311) q[8];
rz(-0.4392906536977774) q[8];
ry(-1.5641492864977489) q[9];
rz(-0.016290009567136114) q[9];
ry(1.5706864080636953) q[10];
rz(3.141461143776692) q[10];
ry(1.5931559285555812) q[11];
rz(1.8352390362590771) q[11];
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
ry(-1.0496902883415204) q[0];
rz(-0.002690790195309489) q[0];
ry(7.38342613032243e-05) q[1];
rz(-1.2788267636511836) q[1];
ry(-2.4811661082775975) q[2];
rz(-0.7361742446204927) q[2];
ry(3.1393893117298868) q[3];
rz(1.5210871523353442) q[3];
ry(3.141528458788164) q[4];
rz(1.2409049627998912) q[4];
ry(0.05262327426090697) q[5];
rz(-2.653114718541977) q[5];
ry(-1.4870308887783794) q[6];
rz(-3.056632215637749) q[6];
ry(-0.19147026013830087) q[7];
rz(-0.012139574819615364) q[7];
ry(0.00012666729664978504) q[8];
rz(2.0097672992645803) q[8];
ry(0.11245812307876957) q[9];
rz(-2.722029884007923) q[9];
ry(1.8710870439626612) q[10];
rz(1.5671918320635898) q[10];
ry(3.141533812767209) q[11];
rz(2.9306493802535387) q[11];
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
ry(-1.6188654216405993) q[0];
rz(-0.05870642018972458) q[0];
ry(-1.5705886174080357) q[1];
rz(1.5718276900740742) q[1];
ry(-1.106262709349437e-05) q[2];
rz(0.7362145693801126) q[2];
ry(1.5722497219670037) q[3];
rz(-1.5711883676814586) q[3];
ry(-0.7939844946357074) q[4];
rz(-1.571501369021072) q[4];
ry(1.9930395190787094) q[5];
rz(-0.024697559907206387) q[5];
ry(3.063253770908165) q[6];
rz(1.580168214481076) q[6];
ry(-1.5611135538961887) q[7];
rz(-0.9794327922986046) q[7];
ry(2.3254660352681324) q[8];
rz(-0.30891834058856665) q[8];
ry(0.051917279275268235) q[9];
rz(2.579026138408852) q[9];
ry(-2.9119703556356606) q[10];
rz(-1.5732601412359308) q[10];
ry(3.141169448625244) q[11];
rz(-2.332069686922016) q[11];
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
ry(1.5716356298506562) q[0];
rz(1.5958870928254605) q[0];
ry(1.570777878864619) q[1];
rz(1.5612701181758482) q[1];
ry(1.4285481678237952) q[2];
rz(1.570810363460697) q[2];
ry(1.5731232698498239) q[3];
rz(-0.9336077530464659) q[3];
ry(-1.6213850379723629) q[4];
rz(-0.1258158905508724) q[4];
ry(3.14158158389981) q[5];
rz(0.9106035868327754) q[5];
ry(3.1414902431382132) q[6];
rz(-0.021981371112424593) q[6];
ry(1.3190599975843043e-05) q[7];
rz(2.5362934178311063) q[7];
ry(3.141560236813952) q[8];
rz(2.8334560208427564) q[8];
ry(3.1413685903002615) q[9];
rz(-0.3370994927397435) q[9];
ry(-2.9505574702943527) q[10];
rz(1.5731211087115904) q[10];
ry(0.00029893365940576104) q[11];
rz(1.8185784968116039) q[11];
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
ry(-1.5551060985283214) q[0];
rz(3.0517344901941788) q[0];
ry(1.5597118056057193) q[1];
rz(0.27215524168792765) q[1];
ry(2.6886550125267745) q[2];
rz(1.039937321699604) q[2];
ry(-2.8889450836936406) q[3];
rz(-0.2726989263520247) q[3];
ry(-3.12909784455577) q[4];
rz(1.4370076308754154) q[4];
ry(-0.013616734268415915) q[5];
rz(-0.9252683614621966) q[5];
ry(-2.885529807661971) q[6];
rz(-1.548277669507458) q[6];
ry(-1.470097995470507) q[7];
rz(1.6026608675955245) q[7];
ry(-1.564163522910776) q[8];
rz(3.141316713044558) q[8];
ry(3.1384820162319684) q[9];
rz(-1.7496729608137338) q[9];
ry(0.6309856342414635) q[10];
rz(1.5691843917176227) q[10];
ry(1.5376665795743403) q[11];
rz(-2.8609979468905795) q[11];
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
ry(-2.8879256528942556) q[0];
rz(0.07263827051567694) q[0];
ry(-3.1191007318954083) q[1];
rz(-1.4879250289736101) q[1];
ry(-3.141376043011836) q[2];
rz(0.9963460629959348) q[2];
ry(1.3108711829493416) q[3];
rz(-1.6205701168211482) q[3];
ry(1.6171413358107947) q[4];
rz(-1.4314881502734584) q[4];
ry(1.545165982110032) q[5];
rz(-2.4678684669554105) q[5];
ry(-1.7605923908437393) q[6];
rz(-0.02568476554886079) q[6];
ry(-1.3809922517698876) q[7];
rz(1.7313078682719623) q[7];
ry(1.5706631416773615) q[8];
rz(-0.41929678712884133) q[8];
ry(1.5770730852985198) q[9];
rz(3.1255333156808307) q[9];
ry(2.6745891071837007) q[10];
rz(-0.17974247645675856) q[10];
ry(-0.000619447261820838) q[11];
rz(2.8622051118645024) q[11];
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
ry(-0.0007539304175186464) q[0];
rz(-0.7316265095125393) q[0];
ry(3.140860927822708) q[1];
rz(-2.8961377204722054) q[1];
ry(0.0022685259443757464) q[2];
rz(-1.5200503390804156) q[2];
ry(1.606307689971942e-05) q[3];
rz(1.4159920364213017) q[3];
ry(3.141476751964854) q[4];
rz(-2.851057899989037) q[4];
ry(3.1415334368403) q[5];
rz(2.2546318059633133) q[5];
ry(-3.1415060416224865) q[6];
rz(-1.5185153153646245) q[6];
ry(-2.1775296983328474e-05) q[7];
rz(2.703204291384627) q[7];
ry(-1.5708616397371467) q[8];
rz(0.9109327737979366) q[8];
ry(-1.5707591878673188) q[9];
rz(1.2509046379021123) q[9];
ry(5.982193176290451e-05) q[10];
rz(0.15103887914733366) q[10];
ry(-1.57299988733861) q[11];
rz(-1.573117619469093) q[11];
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
ry(-3.117054900612228) q[0];
rz(-2.8628712471297937) q[0];
ry(0.027497328998557124) q[1];
rz(-2.4464266722578243) q[1];
ry(-0.8203127438257968) q[2];
rz(1.7990309352723215) q[2];
ry(-1.5662277207508266) q[3];
rz(0.18093392737214395) q[3];
ry(-1.5726004581529434) q[4];
rz(3.025591529932474) q[4];
ry(-1.582813397909236) q[5];
rz(2.6885053642400143) q[5];
ry(1.5502443548230787) q[6];
rz(1.5290117484888992) q[6];
ry(0.21365861652696072) q[7];
rz(-2.809337041632645) q[7];
ry(-3.1372410973028337) q[8];
rz(-0.6995776074846443) q[8];
ry(-3.1076064318257477) q[9];
rz(1.2239561494977318) q[9];
ry(-2.7269163644994356) q[10];
rz(-0.27304442739095774) q[10];
ry(-1.1696390355022084) q[11];
rz(-2.108909958752962) q[11];
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
ry(8.139403465282966e-05) q[0];
rz(-1.696972613070554) q[0];
ry(0.0001317104635398159) q[1];
rz(1.2222641481999954) q[1];
ry(0.001173804118136015) q[2];
rz(1.3482749743484135) q[2];
ry(-3.141554373585765) q[3];
rz(0.18171190738228307) q[3];
ry(-0.0006248591968107675) q[4];
rz(0.5483727614586265) q[4];
ry(0.0007173830485516279) q[5];
rz(-1.1054081079248774) q[5];
ry(3.135237935134145) q[6];
rz(-0.042304615015703984) q[6];
ry(3.0959307271394527) q[7];
rz(-1.4770506164510087) q[7];
ry(3.141186517139529) q[8];
rz(-0.037852256546266866) q[8];
ry(-0.04603578107113127) q[9];
rz(1.6030797886859185) q[9];
ry(-3.1414241196197414) q[10];
rz(-1.8179949726254403) q[10];
ry(-3.1401675636391335) q[11];
rz(-0.5389201336849762) q[11];
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
ry(-3.1075784471899985) q[0];
rz(-2.0726625766438787) q[0];
ry(3.102769328382233) q[1];
rz(-1.6858747830883547) q[1];
ry(2.704753595216739) q[2];
rz(1.5827234875604805) q[2];
ry(1.509123493196629) q[3];
rz(-1.9189685846661364) q[3];
ry(-0.08780237609900046) q[4];
rz(1.7714937550508767) q[4];
ry(1.5302078967460864) q[5];
rz(3.0403025208331487) q[5];
ry(1.61054816449753) q[6];
rz(3.0911819124387563) q[6];
ry(1.6207330620187044) q[7];
rz(1.423692974290782) q[7];
ry(1.5795774864052712) q[8];
rz(2.034843801536069) q[8];
ry(1.6012667800652594) q[9];
rz(1.252048977103177) q[9];
ry(1.5818707769054647) q[10];
rz(1.4625621468557632) q[10];
ry(1.5686160128401125) q[11];
rz(-1.1710557842613314) q[11];
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
ry(-5.6996797949793176e-05) q[0];
rz(-1.6646162688833486) q[0];
ry(-3.141529944476197) q[1];
rz(1.9938999837014728) q[1];
ry(-1.5707394762060791) q[2];
rz(2.948011532830863) q[2];
ry(-3.1414627328758615) q[3];
rz(2.6212357033943325) q[3];
ry(1.9220634243777603e-05) q[4];
rz(2.4122441840644773) q[4];
ry(-3.141583286186229) q[5];
rz(-1.5346491788639478) q[5];
ry(-3.1415015578586214) q[6];
rz(-1.9416200815024665) q[6];
ry(-3.1414995145585944) q[7];
rz(1.575024392636644) q[7];
ry(-0.00016523216580306994) q[8];
rz(0.3869082972846502) q[8];
ry(-3.1414261779601826) q[9];
rz(1.11197175424924) q[9];
ry(1.570842556015088) q[10];
rz(-0.013951103150859906) q[10];
ry(1.571032150903469) q[11];
rz(-2.6687949243708355) q[11];
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
ry(-1.6573556553976012) q[0];
rz(2.8007640429236043) q[0];
ry(1.4689049886678776) q[1];
rz(-0.3340351285716015) q[1];
ry(1.3933453820891346) q[2];
rz(2.4668126841718157) q[2];
ry(1.764359940634356) q[3];
rz(2.1854867485829748) q[3];
ry(1.587402691975137) q[4];
rz(1.2358295667074837) q[4];
ry(-0.0968195047341722) q[5];
rz(2.5721781350532855) q[5];
ry(-0.0011906834300772218) q[6];
rz(2.9626999187224974) q[6];
ry(0.00016615774974759742) q[7];
rz(-0.8425553102884933) q[7];
ry(-3.140660292110734) q[8];
rz(-0.14560714017591803) q[8];
ry(0.004119505621855701) q[9];
rz(-0.8462944250501261) q[9];
ry(1.5979871250042819) q[10];
rz(-2.11194536125958) q[10];
ry(-0.030441261537115138) q[11];
rz(-1.502913445399417) q[11];
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
ry(-0.688979153953948) q[0];
rz(-1.8972597025567088) q[0];
ry(0.6863936584680204) q[1];
rz(1.2354377188650594) q[1];
ry(-0.6089710117858891) q[2];
rz(-1.7461443415375502) q[2];
ry(0.39593050340253755) q[3];
rz(-0.05029255746116258) q[3];
ry(0.6221638958797897) q[4];
rz(1.172906392429771) q[4];
ry(-0.6386022746997937) q[5];
rz(-1.8661861081491915) q[5];
ry(-0.6445571411737266) q[6];
rz(-1.8093036979976302) q[6];
ry(-0.6469651042917008) q[7];
rz(-1.8034559695641086) q[7];
ry(-1.8627311297760791) q[8];
rz(-0.6530796983012178) q[8];
ry(-1.8613865266494558) q[9];
rz(-0.6640388409346669) q[9];
ry(0.337773782559168) q[10];
rz(3.043196579241697) q[10];
ry(2.8038292581444786) q[11];
rz(-0.09824326844755403) q[11];