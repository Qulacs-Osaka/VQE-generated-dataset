OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.5356524162333771) q[0];
ry(-2.458677047035007) q[1];
cx q[0],q[1];
ry(2.669769541570013) q[0];
ry(-0.546677340466955) q[1];
cx q[0],q[1];
ry(1.356877292592728) q[0];
ry(1.1645506497768343) q[2];
cx q[0],q[2];
ry(-2.0946264901921503) q[0];
ry(2.080904112956865) q[2];
cx q[0],q[2];
ry(-0.1957016026334788) q[0];
ry(1.403766554131721) q[3];
cx q[0],q[3];
ry(-2.3343398377427844) q[0];
ry(-0.8704309162350192) q[3];
cx q[0],q[3];
ry(-2.2227938170080517) q[1];
ry(-1.9402986879933364) q[2];
cx q[1],q[2];
ry(0.05028783938969149) q[1];
ry(-2.0354691725407488) q[2];
cx q[1],q[2];
ry(-0.1718431774166449) q[1];
ry(0.8844896953693819) q[3];
cx q[1],q[3];
ry(-0.6524180756058108) q[1];
ry(-1.1483664597488143) q[3];
cx q[1],q[3];
ry(-2.4769997743636174) q[2];
ry(-0.03022370143265376) q[3];
cx q[2],q[3];
ry(-1.7461234260568408) q[2];
ry(-2.715781868030205) q[3];
cx q[2],q[3];
ry(0.7491676335641779) q[0];
ry(0.6567813672352703) q[1];
cx q[0],q[1];
ry(1.036055852115527) q[0];
ry(-0.6946266408981812) q[1];
cx q[0],q[1];
ry(-0.3221976275177517) q[0];
ry(-2.596424468628585) q[2];
cx q[0],q[2];
ry(-1.2328379962623872) q[0];
ry(-2.5611320224893532) q[2];
cx q[0],q[2];
ry(-3.013022899044181) q[0];
ry(0.8981734438304727) q[3];
cx q[0],q[3];
ry(-2.0216468029065595) q[0];
ry(-1.1022110276438886) q[3];
cx q[0],q[3];
ry(-2.661111848817769) q[1];
ry(-2.1728421924274386) q[2];
cx q[1],q[2];
ry(2.0346040531974) q[1];
ry(1.7208751729894485) q[2];
cx q[1],q[2];
ry(-0.5274503578657822) q[1];
ry(1.0161894170321064) q[3];
cx q[1],q[3];
ry(3.0711795476362505) q[1];
ry(-0.5334200492374702) q[3];
cx q[1],q[3];
ry(1.9589377999503448) q[2];
ry(1.685409631953517) q[3];
cx q[2],q[3];
ry(-2.6146750877868787) q[2];
ry(3.0506512227312035) q[3];
cx q[2],q[3];
ry(2.57319253719765) q[0];
ry(-1.317983638472799) q[1];
cx q[0],q[1];
ry(0.29404834673970837) q[0];
ry(1.2609073750175082) q[1];
cx q[0],q[1];
ry(-1.9801565080936436) q[0];
ry(-2.064818837505579) q[2];
cx q[0],q[2];
ry(-2.007041436011172) q[0];
ry(-1.6163052562687519) q[2];
cx q[0],q[2];
ry(-0.18999293613226786) q[0];
ry(1.393233708366431) q[3];
cx q[0],q[3];
ry(1.964984264896292) q[0];
ry(-2.183529624794871) q[3];
cx q[0],q[3];
ry(1.1366797810258253) q[1];
ry(-1.3522091772393574) q[2];
cx q[1],q[2];
ry(1.8149518710093826) q[1];
ry(2.554022952634376) q[2];
cx q[1],q[2];
ry(-2.973502739226715) q[1];
ry(-0.8340217918377828) q[3];
cx q[1],q[3];
ry(-2.705525650921183) q[1];
ry(0.6894095705514905) q[3];
cx q[1],q[3];
ry(-0.47936762095342544) q[2];
ry(1.8368273899503604) q[3];
cx q[2],q[3];
ry(1.0312492570976524) q[2];
ry(-0.4097707306887085) q[3];
cx q[2],q[3];
ry(0.22070816331046098) q[0];
ry(2.9883992189142896) q[1];
cx q[0],q[1];
ry(-0.6924634569760569) q[0];
ry(1.261415234317725) q[1];
cx q[0],q[1];
ry(-2.1642640024119224) q[0];
ry(1.1565612232648945) q[2];
cx q[0],q[2];
ry(1.7932636357292893) q[0];
ry(-3.0686254972751192) q[2];
cx q[0],q[2];
ry(-0.4212796135927654) q[0];
ry(0.217054422122615) q[3];
cx q[0],q[3];
ry(-2.5360240900006854) q[0];
ry(-1.6249778068685776) q[3];
cx q[0],q[3];
ry(-2.634755982714826) q[1];
ry(0.0032829201023023558) q[2];
cx q[1],q[2];
ry(-0.5522258024410478) q[1];
ry(-2.319898472441737) q[2];
cx q[1],q[2];
ry(0.17962402810478426) q[1];
ry(-2.4118867461866285) q[3];
cx q[1],q[3];
ry(-1.282134887606733) q[1];
ry(2.867926662060692) q[3];
cx q[1],q[3];
ry(-1.240076009598378) q[2];
ry(-1.8021598825715914) q[3];
cx q[2],q[3];
ry(1.2478612479597988) q[2];
ry(2.658202224582001) q[3];
cx q[2],q[3];
ry(-1.1004369477992144) q[0];
ry(0.2004982100044881) q[1];
cx q[0],q[1];
ry(0.9252167136018494) q[0];
ry(1.2582975618653043) q[1];
cx q[0],q[1];
ry(-2.9899310881764447) q[0];
ry(-0.08938623135840107) q[2];
cx q[0],q[2];
ry(2.625505813226896) q[0];
ry(-1.7703151661690935) q[2];
cx q[0],q[2];
ry(1.1622055682690613) q[0];
ry(2.9660134872139934) q[3];
cx q[0],q[3];
ry(1.5517193339555757) q[0];
ry(1.53434003413797) q[3];
cx q[0],q[3];
ry(-1.9324375914635166) q[1];
ry(-0.8208720141955086) q[2];
cx q[1],q[2];
ry(0.1021730721345499) q[1];
ry(0.9295217485261933) q[2];
cx q[1],q[2];
ry(-3.126585995313078) q[1];
ry(2.5893827562643352) q[3];
cx q[1],q[3];
ry(2.5787837312271034) q[1];
ry(1.0806042749140943) q[3];
cx q[1],q[3];
ry(-1.6024901228339061) q[2];
ry(-2.454353842471133) q[3];
cx q[2],q[3];
ry(-3.088776861546918) q[2];
ry(0.9858724626111305) q[3];
cx q[2],q[3];
ry(-3.0064895265815426) q[0];
ry(-1.765777447087804) q[1];
cx q[0],q[1];
ry(1.3278899828700492) q[0];
ry(2.106335408792634) q[1];
cx q[0],q[1];
ry(-1.3064391551439494) q[0];
ry(2.9687035985388763) q[2];
cx q[0],q[2];
ry(-1.5876338713483866) q[0];
ry(-0.7202481937107121) q[2];
cx q[0],q[2];
ry(2.769205210145066) q[0];
ry(1.0545779141047307) q[3];
cx q[0],q[3];
ry(-1.6255736421789437) q[0];
ry(-1.7250043933346069) q[3];
cx q[0],q[3];
ry(0.3462647587721497) q[1];
ry(1.6218058729365987) q[2];
cx q[1],q[2];
ry(2.8311116699936445) q[1];
ry(-0.74925540067255) q[2];
cx q[1],q[2];
ry(-1.804663600924008) q[1];
ry(-1.7194857454606742) q[3];
cx q[1],q[3];
ry(0.724745314622619) q[1];
ry(-1.0098378094084144) q[3];
cx q[1],q[3];
ry(1.1257680392567189) q[2];
ry(2.9448484113558084) q[3];
cx q[2],q[3];
ry(2.345070678700473) q[2];
ry(-2.6478800921154995) q[3];
cx q[2],q[3];
ry(0.41561256336732594) q[0];
ry(-1.873098194182796) q[1];
ry(-2.858208321913352) q[2];
ry(0.9734992617880106) q[3];