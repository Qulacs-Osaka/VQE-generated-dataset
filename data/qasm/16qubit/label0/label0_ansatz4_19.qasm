OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(0.0710069789325507) q[0];
rz(0.07276440725722556) q[0];
ry(0.10450020582297093) q[1];
rz(2.8133694204576556) q[1];
ry(-1.6033333641674699) q[2];
rz(-2.9364874350114007) q[2];
ry(-1.4579347161034377) q[3];
rz(0.9227251032496788) q[3];
ry(0.00029583503459289836) q[4];
rz(-0.5755014399713316) q[4];
ry(-1.794656684075458e-05) q[5];
rz(-1.3339992271498151) q[5];
ry(1.554089072294415) q[6];
rz(-0.1948982593987032) q[6];
ry(-1.5279475812662275) q[7];
rz(1.6138053955607994) q[7];
ry(0.0004111080584472405) q[8];
rz(-1.2598205570254892) q[8];
ry(-0.0005681855784054335) q[9];
rz(-2.9716607106720816) q[9];
ry(1.570677018489916) q[10];
rz(0.9054055091215032) q[10];
ry(1.5756575431976234) q[11];
rz(0.23754907729436334) q[11];
ry(3.1414414840419225) q[12];
rz(2.427187471697889) q[12];
ry(3.1391602701696413) q[13];
rz(1.0774904307128264) q[13];
ry(2.7131602325382445) q[14];
rz(-1.5811519275092198) q[14];
ry(-3.131039270217701) q[15];
rz(1.7019400913120055) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.034862242538344856) q[0];
rz(-1.2611254026558854) q[0];
ry(1.1236466522582624) q[1];
rz(-0.7278652616091275) q[1];
ry(1.484192205809495) q[2];
rz(1.8231500306191797) q[2];
ry(1.8415957640427925) q[3];
rz(-1.7497019132987504) q[3];
ry(-3.141393112084405) q[4];
rz(-0.7104391874062848) q[4];
ry(-3.1414789462054986) q[5];
rz(-1.8377038339646214) q[5];
ry(-3.1156220658666425) q[6];
rz(-1.6439991128824483) q[6];
ry(3.037203117733075) q[7];
rz(-0.047696997506805124) q[7];
ry(2.2198605048227003) q[8];
rz(-1.4352490834936336) q[8];
ry(3.1154051105410625) q[9];
rz(-1.1914848369608773) q[9];
ry(-0.29567605968378996) q[10];
rz(-2.7633449769235434) q[10];
ry(-1.3881004820412017) q[11];
rz(-1.31122075156589) q[11];
ry(-1.6543127082406706) q[12];
rz(-1.0047446424750028) q[12];
ry(1.299194328437073) q[13];
rz(2.674298429091426) q[13];
ry(1.1011741374254926) q[14];
rz(-0.9007593315148954) q[14];
ry(0.05115094160171979) q[15];
rz(0.2626147959909414) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.0985678348449455) q[0];
rz(-0.9048882884252006) q[0];
ry(3.1081195269234723) q[1];
rz(-1.5851140290074417) q[1];
ry(-3.101233164283534) q[2];
rz(2.1016667328672494) q[2];
ry(1.8238880381135425) q[3];
rz(2.698402618890697) q[3];
ry(0.00033507252676923116) q[4];
rz(1.0834761428718105) q[4];
ry(1.6538747106305352e-05) q[5];
rz(2.953703332506233) q[5];
ry(-0.6815917170908117) q[6];
rz(2.002105310751658) q[6];
ry(-1.4127055835238913) q[7];
rz(1.749294662749297) q[7];
ry(-3.135516249848308) q[8];
rz(2.599929620204356) q[8];
ry(0.012217212038968308) q[9];
rz(2.2757948372012455) q[9];
ry(2.984437057454453) q[10];
rz(-0.5689473368176843) q[10];
ry(0.1549422519471667) q[11];
rz(2.575709532472451) q[11];
ry(1.369966791660558) q[12];
rz(0.6458015431102258) q[12];
ry(1.41913569069263) q[13];
rz(-3.0711460744559664) q[13];
ry(1.528891399622462) q[14];
rz(-1.4871235703275028) q[14];
ry(-3.026465535873041) q[15];
rz(-0.25397167501291396) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.776637745873154) q[0];
rz(-0.7912598461686207) q[0];
ry(2.6874335097479953) q[1];
rz(2.528146789318717) q[1];
ry(-1.9618339647112208) q[2];
rz(-1.4726126714495473) q[2];
ry(-2.601874853065134) q[3];
rz(-0.6582698020029334) q[3];
ry(0.3240980849941456) q[4];
rz(0.529249710179081) q[4];
ry(0.3048074277361881) q[5];
rz(-1.8330435702207026) q[5];
ry(-2.874146358870821) q[6];
rz(-0.8225944967969836) q[6];
ry(-1.3522152880776863) q[7];
rz(-0.5980090780136549) q[7];
ry(-1.7797819631192462) q[8];
rz(3.100728513739634) q[8];
ry(3.130035966885797) q[9];
rz(2.8905835517543115) q[9];
ry(-2.1914091345087092) q[10];
rz(-2.771973492158883) q[10];
ry(0.9479681277773215) q[11];
rz(-0.3686106755144624) q[11];
ry(1.9514133797792577) q[12];
rz(2.835090769645598) q[12];
ry(0.6463398636116393) q[13];
rz(-1.8319299538990341) q[13];
ry(-2.5380205704308216) q[14];
rz(-2.0541457266702103) q[14];
ry(0.01809262741508692) q[15];
rz(-1.274163759419036) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.8236601186043557) q[0];
rz(2.9047516585185957) q[0];
ry(0.32675484401685934) q[1];
rz(-1.3348519515012023) q[1];
ry(-0.00153335483070044) q[2];
rz(2.242081484425675) q[2];
ry(0.003452234907816324) q[3];
rz(-1.511993941214299) q[3];
ry(-0.08439356046723034) q[4];
rz(2.577026484075945) q[4];
ry(3.129894430348088) q[5];
rz(-0.5876484717729564) q[5];
ry(-3.1401179893888367) q[6];
rz(-1.998215099093664) q[6];
ry(0.0024657581357860896) q[7];
rz(0.7921153378512572) q[7];
ry(1.1297392257554932) q[8];
rz(-2.68951004366036) q[8];
ry(1.3977642935316181) q[9];
rz(-0.08375830224459424) q[9];
ry(-1.4715679143899205) q[10];
rz(0.9627094484947731) q[10];
ry(-1.4734867029710914) q[11];
rz(-0.3153671184072155) q[11];
ry(1.2718732537835395) q[12];
rz(-2.484797323861179) q[12];
ry(1.8800133099471719) q[13];
rz(2.3726223150452515) q[13];
ry(-0.40260092489183735) q[14];
rz(-1.1922625251696592) q[14];
ry(-2.7762404895861046) q[15];
rz(-1.6708399759020298) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.0639680279562933) q[0];
rz(1.4240749490218376) q[0];
ry(-1.6952870994012967) q[1];
rz(1.6284713356142457) q[1];
ry(0.0068315896477697535) q[2];
rz(2.5612017556252686) q[2];
ry(-3.1220385808488595) q[3];
rz(0.20417858437902064) q[3];
ry(-0.03698592604783927) q[4];
rz(2.050528654750486) q[4];
ry(-0.8193134133646627) q[5];
rz(2.288193561078753) q[5];
ry(3.1395363778756886) q[6];
rz(-0.11946617491975431) q[6];
ry(0.03792979261140417) q[7];
rz(-2.3407882183683655) q[7];
ry(3.0925923785423994) q[8];
rz(0.029790865729219188) q[8];
ry(1.2729871012492668) q[9];
rz(0.5667983132638004) q[9];
ry(-3.1401466637170654) q[10];
rz(-0.8888328372358343) q[10];
ry(-0.0002953230332190206) q[11];
rz(-0.9173273232825859) q[11];
ry(-0.26790426067490986) q[12];
rz(0.0036850508290546817) q[12];
ry(-0.16889478592698914) q[13];
rz(-2.5886774335099836) q[13];
ry(-0.30346219736403857) q[14];
rz(2.01569030816113) q[14];
ry(-3.0917104193516356) q[15];
rz(2.693287392004271) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.6075856772603778) q[0];
rz(-0.7768796867119595) q[0];
ry(-2.43108711117321) q[1];
rz(-2.68626071512733) q[1];
ry(1.567922574423482) q[2];
rz(-2.5211125863295996) q[2];
ry(-1.5760731791098102) q[3];
rz(2.3720236347302865) q[3];
ry(-2.915758590588825) q[4];
rz(0.15447422604294161) q[4];
ry(2.404281433359901) q[5];
rz(0.8939214107666604) q[5];
ry(-0.009815645043449579) q[6];
rz(0.9640594935094232) q[6];
ry(-0.019904920831101514) q[7];
rz(3.1108012488593344) q[7];
ry(0.06091233017053454) q[8];
rz(-1.1170224392283359) q[8];
ry(-1.7728065292157407) q[9];
rz(0.10804905132556854) q[9];
ry(-3.140095721274504) q[10];
rz(-2.8744513716817455) q[10];
ry(-0.0011254224431148224) q[11];
rz(3.1157855752370147) q[11];
ry(-3.0488146705406325) q[12];
rz(2.0980348980160533) q[12];
ry(2.9347594889208777) q[13];
rz(2.0466318039085953) q[13];
ry(0.959023236175268) q[14];
rz(1.801982083230862) q[14];
ry(-0.04483924322844552) q[15];
rz(2.6766648128263197) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.3170677103164734) q[0];
rz(-1.6971185402326396) q[0];
ry(-2.2949182410173665) q[1];
rz(1.7325934798019702) q[1];
ry(-3.137510490530162) q[2];
rz(-0.9436404743079726) q[2];
ry(-0.002083482619561984) q[3];
rz(2.327738256005156) q[3];
ry(3.1395484171381836) q[4];
rz(-2.4685736658201582) q[4];
ry(-3.12176372478467) q[5];
rz(0.8929235495610117) q[5];
ry(-3.1282107094015643) q[6];
rz(-1.352990005729268) q[6];
ry(0.009689453841256856) q[7];
rz(2.4290426908039318) q[7];
ry(-0.02646415229312815) q[8];
rz(-2.464061545972927) q[8];
ry(-1.899294538663323) q[9];
rz(-0.6248657289232595) q[9];
ry(0.0001071725827247389) q[10];
rz(2.38789348449857) q[10];
ry(0.00037503979537500953) q[11];
rz(-0.09599713519624586) q[11];
ry(-1.8403807263032788) q[12];
rz(0.024890586003893513) q[12];
ry(1.5672353830897856) q[13];
rz(2.603161244253467) q[13];
ry(-2.5651155862869173) q[14];
rz(1.587389825700885) q[14];
ry(-0.1954615293832064) q[15];
rz(-0.2855896588820901) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.8126427357704435) q[0];
rz(0.5554833520367349) q[0];
ry(-0.7610676425540417) q[1];
rz(-0.013318848936791028) q[1];
ry(-0.0009021839318977249) q[2];
rz(-1.5851076375000694) q[2];
ry(-3.0999670128452284) q[3];
rz(1.5587642187861477) q[3];
ry(-0.37098831802068233) q[4];
rz(0.9552910775645936) q[4];
ry(-0.7724728326157164) q[5];
rz(-1.9748775655044604) q[5];
ry(-3.1342390258612935) q[6];
rz(-1.3536973541847352) q[6];
ry(-3.126992216680763) q[7];
rz(3.1335120023647627) q[7];
ry(-0.08406291315019665) q[8];
rz(0.7006866521158656) q[8];
ry(-0.07884747951497052) q[9];
rz(-2.8492958117236404) q[9];
ry(-0.6164798893107698) q[10];
rz(-0.4298309461939646) q[10];
ry(2.4893470355589855) q[11];
rz(2.155335928960982) q[11];
ry(-2.679133190610429) q[12];
rz(-2.8276015773323433) q[12];
ry(0.8492564457830902) q[13];
rz(-0.5250416074846552) q[13];
ry(0.7896465602721133) q[14];
rz(-0.8848719700299776) q[14];
ry(-1.493716416085217) q[15];
rz(0.07054401548631556) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.734919142863987) q[0];
rz(-2.9477997508154075) q[0];
ry(1.962091796665119) q[1];
rz(1.426463950345326) q[1];
ry(-1.5735660870127006) q[2];
rz(3.028299805311653) q[2];
ry(1.579093091020052) q[3];
rz(-2.464666433104618) q[3];
ry(3.137880272615667) q[4];
rz(-0.03716496999943683) q[4];
ry(-0.005058161626942059) q[5];
rz(-0.7125363337353764) q[5];
ry(0.2938517774572266) q[6];
rz(1.9741231718032557) q[6];
ry(0.07277135848912201) q[7];
rz(0.30234670510156) q[7];
ry(-1.6663385569441826) q[8];
rz(-2.174362094787605) q[8];
ry(3.134300572715677) q[9];
rz(-2.922060025277195) q[9];
ry(1.3343508849188495) q[10];
rz(1.3886539312195205) q[10];
ry(-2.392842783211177) q[11];
rz(1.2348255187544592) q[11];
ry(0.15051881051569982) q[12];
rz(-1.597985091890034) q[12];
ry(-0.02603912517909497) q[13];
rz(2.431823399904128) q[13];
ry(1.5472716610268664) q[14];
rz(0.8954158725353427) q[14];
ry(0.4470927495644599) q[15];
rz(0.09375153415796154) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.9048412682227127) q[0];
rz(-1.0727736671082084) q[0];
ry(1.1109501208626331) q[1];
rz(-1.9021087640665046) q[1];
ry(-0.05800177628263814) q[2];
rz(2.7137098395447503) q[2];
ry(1.5584671613256358) q[3];
rz(-1.0770802627568408) q[3];
ry(-0.09102577081794827) q[4];
rz(0.8555794263059777) q[4];
ry(-0.7861815773545162) q[5];
rz(-2.0466042399722033) q[5];
ry(-2.9806991489333003) q[6];
rz(-0.8701827097483933) q[6];
ry(-1.6084214096374012) q[7];
rz(-0.16095624423896823) q[7];
ry(3.1394198871131818) q[8];
rz(-1.6679830812982264) q[8];
ry(-0.0012349706113771249) q[9];
rz(-1.5702149971366188) q[9];
ry(-0.309318901118906) q[10];
rz(2.9801461278444386) q[10];
ry(1.0782893235257491) q[11];
rz(-1.1091444374852077) q[11];
ry(0.2999734249226673) q[12];
rz(-0.24203029005805465) q[12];
ry(-0.4182470148006309) q[13];
rz(-2.6280625464838065) q[13];
ry(1.737882766925147) q[14];
rz(-1.2433259626643052) q[14];
ry(0.4600132065657574) q[15];
rz(0.33306073225537625) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.2257810154432773) q[0];
rz(-1.1984380063714362) q[0];
ry(2.934464875456458) q[1];
rz(-1.211509842717092) q[1];
ry(-1.4745267219792975) q[2];
rz(-1.1562218091564436) q[2];
ry(2.810485118417955) q[3];
rz(0.5178085945539825) q[3];
ry(3.1413232385993877) q[4];
rz(2.26086748115011) q[4];
ry(-3.140986575867401) q[5];
rz(-1.0770620894738194) q[5];
ry(2.205232875320181) q[6];
rz(-1.0555427062657086) q[6];
ry(-0.5303043011082238) q[7];
rz(1.0591671974014663) q[7];
ry(-0.005865700103562688) q[8];
rz(-0.18423959584721494) q[8];
ry(3.140849085171286) q[9];
rz(0.5988754302494704) q[9];
ry(1.7888267485681295) q[10];
rz(-1.3547755103801826) q[10];
ry(-2.5080764537427194) q[11];
rz(-2.4109483553230033) q[11];
ry(3.1155315193400694) q[12];
rz(-1.8861384601491222) q[12];
ry(-3.1186402810381932) q[13];
rz(-1.0831203413303543) q[13];
ry(-1.8488997344667122) q[14];
rz(1.610795232991399) q[14];
ry(-1.0672614644601301) q[15];
rz(-1.1577267935676674) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.0574040709119763) q[0];
rz(1.698818343825898) q[0];
ry(-2.9185144891591737) q[1];
rz(-2.716669396519256) q[1];
ry(1.4468018586368903) q[2];
rz(0.33646607527162503) q[2];
ry(-0.5695236140811513) q[3];
rz(-1.6382396061371274) q[3];
ry(-3.1401392015698444) q[4];
rz(1.1225109421358397) q[4];
ry(-3.140432825500398) q[5];
rz(0.14264254929585804) q[5];
ry(0.23953638739714125) q[6];
rz(1.280885362047893) q[6];
ry(3.0768459626536204) q[7];
rz(2.3832662271594836) q[7];
ry(-3.1404059582295827) q[8];
rz(0.5276605688795054) q[8];
ry(3.1408980674262676) q[9];
rz(2.6176832905357714) q[9];
ry(1.238526305525201) q[10];
rz(-1.7656919234375483) q[10];
ry(1.5996991154697071) q[11];
rz(3.027323043698694) q[11];
ry(-1.8090868963718547) q[12];
rz(1.5201397176141123) q[12];
ry(-1.5008344702575018) q[13];
rz(1.97270711105464) q[13];
ry(1.4037230573350838) q[14];
rz(1.6974494495869303) q[14];
ry(2.990206708623884) q[15];
rz(-1.9692174392222643) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.6258271844644296) q[0];
rz(-1.3408196380200952) q[0];
ry(-1.735266276855297) q[1];
rz(-1.6489012403441847) q[1];
ry(-1.6167035426241858) q[2];
rz(-1.3999966603015108) q[2];
ry(1.6773854818641545) q[3];
rz(-1.6495315125582417) q[3];
ry(3.14129229102214) q[4];
rz(-3.09554561886913) q[4];
ry(-0.0019941480962087695) q[5];
rz(1.7361263534209517) q[5];
ry(0.3593121049880601) q[6];
rz(1.062241600899311) q[6];
ry(-1.7198849328736792) q[7];
rz(-2.463625068034509) q[7];
ry(-0.616181553853953) q[8];
rz(-2.688934238330504) q[8];
ry(1.5133384995921537) q[9];
rz(2.557953159464643) q[9];
ry(-3.0245241875325344) q[10];
rz(0.5534685694060931) q[10];
ry(1.536103781425352) q[11];
rz(-1.5543521175627628) q[11];
ry(-1.867931703250066) q[12];
rz(2.1472124899350935) q[12];
ry(-0.996344588973904) q[13];
rz(-0.5471151140575392) q[13];
ry(1.7112952990672188) q[14];
rz(-2.663382733910976) q[14];
ry(-3.029088342816786) q[15];
rz(-1.6811175832569463) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.009348223285707391) q[0];
rz(-2.5871253122122346) q[0];
ry(-1.6801292096621676) q[1];
rz(2.7239061443223296) q[1];
ry(-1.2196263926467987) q[2];
rz(-0.15656958070786833) q[2];
ry(-1.430763370737627) q[3];
rz(-2.33455975976653) q[3];
ry(-3.1410408122593876) q[4];
rz(0.501369039117851) q[4];
ry(3.1406151093493286) q[5];
rz(-0.3306691285096377) q[5];
ry(0.0105718402542529) q[6];
rz(0.11311121809758241) q[6];
ry(-0.014763530179661742) q[7];
rz(0.03970189545994491) q[7];
ry(1.9689146149918226) q[8];
rz(-1.5595503467305187) q[8];
ry(-3.1407037599830043) q[9];
rz(2.7943310123996996) q[9];
ry(-0.0002961459049020121) q[10];
rz(1.0938129907630811) q[10];
ry(0.00018191737112616124) q[11];
rz(-0.9962964279972953) q[11];
ry(-0.0013296757468770732) q[12];
rz(2.9625014735281443) q[12];
ry(-0.0036034266282536365) q[13];
rz(2.4761110179111925) q[13];
ry(3.0152498544972093) q[14];
rz(1.196821390508308) q[14];
ry(0.3504521307501057) q[15];
rz(-0.4451044904097338) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.3673907105548447) q[0];
rz(1.3553685017445387) q[0];
ry(3.1095548235740935) q[1];
rz(1.6013495589987023) q[1];
ry(-3.1246364196782785) q[2];
rz(1.2913911942553153) q[2];
ry(-0.012829791477288133) q[3];
rz(-2.1762569266175564) q[3];
ry(-3.1379186890709083) q[4];
rz(1.8570041033367382) q[4];
ry(3.1369902750596474) q[5];
rz(0.7866232296229255) q[5];
ry(-3.140479346525309) q[6];
rz(-1.7608557025381755) q[6];
ry(-3.141282782317407) q[7];
rz(-1.9440960768735316) q[7];
ry(-1.5724777053352925) q[8];
rz(-2.581326177461647) q[8];
ry(1.565160804840537) q[9];
rz(-0.09885192374287032) q[9];
ry(3.073737514018243) q[10];
rz(-0.7307483730590326) q[10];
ry(0.007537068538417497) q[11];
rz(-3.0731320726700635) q[11];
ry(0.7188022899465247) q[12];
rz(-2.9633397888404525) q[12];
ry(-1.0270167136936106) q[13];
rz(-1.5959940194612434) q[13];
ry(-0.18826041334637722) q[14];
rz(1.6824107608492413) q[14];
ry(-1.0638712601917952) q[15];
rz(1.1210342969568545) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.3037522905099945) q[0];
rz(-0.8839810373553569) q[0];
ry(-1.0013582322221586) q[1];
rz(-0.21957492184800506) q[1];
ry(-1.5376912872485509) q[2];
rz(-1.3447730301182865) q[2];
ry(-1.5328686148129391) q[3];
rz(2.0519329437926555) q[3];
ry(3.1393839143080062) q[4];
rz(1.4558401078237038) q[4];
ry(-3.1408227622377023) q[5];
rz(2.030703925308477) q[5];
ry(-0.3143206176918554) q[6];
rz(0.024735278864043586) q[6];
ry(1.5690214820060735) q[7];
rz(-1.5277274870857191) q[7];
ry(-1.5708719801863147) q[8];
rz(1.4470055974125025) q[8];
ry(-1.3683821639924767) q[9];
rz(-2.9617240219429206) q[9];
ry(0.0003981894801814434) q[10];
rz(-0.25423248260781184) q[10];
ry(-3.141149704016095) q[11];
rz(-1.422886040931361) q[11];
ry(-1.637467192406444) q[12];
rz(-3.1412233998576053) q[12];
ry(1.5993888214263157) q[13];
rz(-0.038779675335741365) q[13];
ry(2.302439769727368) q[14];
rz(-1.9359730170785543) q[14];
ry(-2.7843428071129126) q[15];
rz(-2.6025611255713477) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.2112438016567015) q[0];
rz(0.5617236076801396) q[0];
ry(-0.060202217910504174) q[1];
rz(0.2686162712946366) q[1];
ry(-1.9104732768686778) q[2];
rz(2.951855688670224) q[2];
ry(2.9164770058982854) q[3];
rz(0.7274726091435904) q[3];
ry(0.0030405072871707195) q[4];
rz(-1.2949564399427267) q[4];
ry(-0.00946361339604973) q[5];
rz(-1.1157678550756527) q[5];
ry(1.822367540852208) q[6];
rz(-0.010250574989656456) q[6];
ry(3.1378031939784568) q[7];
rz(1.5070581928169435) q[7];
ry(-3.13975738547909) q[8];
rz(-0.12312369502045861) q[8];
ry(3.1401936296494193) q[9];
rz(-1.5135667670036579) q[9];
ry(-6.152530765466271e-05) q[10];
rz(-0.8499823587861409) q[10];
ry(3.1414338475249934) q[11];
rz(-0.6356006521332738) q[11];
ry(-1.6241524056962389) q[12];
rz(2.6036535806297776) q[12];
ry(1.618654281682659) q[13];
rz(0.6893030797645545) q[13];
ry(-1.6427071319152322) q[14];
rz(1.1204633521491072) q[14];
ry(0.8213023792101425) q[15];
rz(3.0002374631242428) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.07397610440334391) q[0];
rz(-1.0580294699345991) q[0];
ry(1.0397519389902932) q[1];
rz(-0.762605006755087) q[1];
ry(-2.9605958447639846) q[2];
rz(1.3569793722697112) q[2];
ry(0.04968532905177552) q[3];
rz(0.7836297388312446) q[3];
ry(3.141548449499295) q[4];
rz(-2.114629889546462) q[4];
ry(6.103677359177585e-05) q[5];
rz(0.7625680537856123) q[5];
ry(2.827230805670737) q[6];
rz(-0.012352244262957333) q[6];
ry(1.5706607710707061) q[7];
rz(-3.136958085344675) q[7];
ry(1.480234132841069) q[8];
rz(3.0596328268925053) q[8];
ry(3.1404163877661304) q[9];
rz(-1.2777209295546275) q[9];
ry(3.1408830748110055) q[10];
rz(2.1002942354944354) q[10];
ry(-3.1415091545415126) q[11];
rz(1.9911927457384049) q[11];
ry(1.8426841376302103) q[12];
rz(-0.9197621857145799) q[12];
ry(1.3788502619440806) q[13];
rz(-2.1655703166167344) q[13];
ry(-2.105825216932297) q[14];
rz(2.541711289064952) q[14];
ry(-0.8336334380657026) q[15];
rz(-1.4999192004313844) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.27036573360964145) q[0];
rz(-0.9328809253396795) q[0];
ry(-2.0339667716887937) q[1];
rz(0.380500097362521) q[1];
ry(1.6035561517363277) q[2];
rz(2.6114743213841405) q[2];
ry(1.7338847941870057) q[3];
rz(-1.4518604359908358) q[3];
ry(-0.0038320522205501314) q[4];
rz(2.49140934324422) q[4];
ry(-0.008296354995041888) q[5];
rz(-1.4900244635525555) q[5];
ry(-1.571336824309057) q[6];
rz(1.5696440221802639) q[6];
ry(-1.573398933284908) q[7];
rz(-3.005582458830822) q[7];
ry(-3.112371754285681) q[8];
rz(3.060693300699874) q[8];
ry(-1.8229897005632907) q[9];
rz(-2.9874869435240763) q[9];
ry(3.141044035349759) q[10];
rz(-0.28408581495630625) q[10];
ry(0.0002381501107793227) q[11];
rz(-1.5705616330800467) q[11];
ry(1.5341903361503224) q[12];
rz(1.605515881219358) q[12];
ry(1.5055305962910206) q[13];
rz(-1.5942218744824863) q[13];
ry(-1.762557694197227) q[14];
rz(-0.3884137173494257) q[14];
ry(2.3948128287895285) q[15];
rz(-2.634276984854115) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.4171524295541458) q[0];
rz(1.60790738346234) q[0];
ry(0.21125052403879252) q[1];
rz(-0.03598392177126808) q[1];
ry(0.15732741954148197) q[2];
rz(2.3841626772670432) q[2];
ry(2.9729883575836094) q[3];
rz(-1.3886871747954883) q[3];
ry(1.2436327624150274) q[4];
rz(-0.7969285074673442) q[4];
ry(1.7856276230783665) q[5];
rz(-0.23709343133275596) q[5];
ry(1.5871556796532866) q[6];
rz(2.055639230222477) q[6];
ry(3.082308319602797) q[7];
rz(-1.5106883986067077) q[7];
ry(-2.1702689839798337) q[8];
rz(1.5709695636836338) q[8];
ry(3.1051771562277604) q[9];
rz(1.7159512380054205) q[9];
ry(-3.1403628371349974) q[10];
rz(-1.969074164327292) q[10];
ry(-3.1414807350429474) q[11];
rz(1.774359077941182) q[11];
ry(1.3130463141490152) q[12];
rz(1.4865503342749706) q[12];
ry(-1.735515061392203) q[13];
rz(1.4930096207757524) q[13];
ry(-0.2233278182186507) q[14];
rz(1.630779308965713) q[14];
ry(0.5364957711341471) q[15];
rz(2.2193940603927227) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5423425107973587) q[0];
rz(-1.5759598839686646) q[0];
ry(-3.0894241393457995) q[1];
rz(1.970281504876608) q[1];
ry(3.1409743632866465) q[2];
rz(1.7942196180911696) q[2];
ry(-0.0013025892239744152) q[3];
rz(0.7230364778069669) q[3];
ry(-0.003288205409676739) q[4];
rz(2.499626457339251) q[4];
ry(-3.1395402492617106) q[5];
rz(-1.8331201488242854) q[5];
ry(-0.00017751050473640954) q[6];
rz(2.639150933677829) q[6];
ry(3.141519461206368) q[7];
rz(1.491484402905244) q[7];
ry(1.571003063742103) q[8];
rz(-1.5081629631786475) q[8];
ry(1.5691979717644418) q[9];
rz(-1.7468725118966575) q[9];
ry(-4.436378142624874e-05) q[10];
rz(-2.3914933701697034) q[10];
ry(-3.141577996069917) q[11];
rz(2.793082092881191) q[11];
ry(1.6017148200310827) q[12];
rz(-2.6060220103701015) q[12];
ry(1.5448549102165772) q[13];
rz(-0.41004780082334497) q[13];
ry(-1.285119179822341) q[14];
rz(0.012653548345474164) q[14];
ry(-0.2453232345130027) q[15];
rz(1.4844193772380745) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.6319549038213474) q[0];
rz(-1.3891279180556648) q[0];
ry(2.1729603707861362) q[1];
rz(-1.7617074443312943) q[1];
ry(1.5472665622203357) q[2];
rz(0.17995303626692968) q[2];
ry(1.443006896679143) q[3];
rz(3.1298558090068704) q[3];
ry(0.7018381716963713) q[4];
rz(1.2587516315098861) q[4];
ry(-2.8988099472209132) q[5];
rz(-1.675747411428607) q[5];
ry(1.5979961341988167) q[6];
rz(1.6034722977665954) q[6];
ry(1.5826443254278901) q[7];
rz(1.5573363293831166) q[7];
ry(3.1376464076922237) q[8];
rz(-1.51069995571822) q[8];
ry(-0.6296364440242073) q[9];
rz(-1.400987066274328) q[9];
ry(-1.620046967166865) q[10];
rz(-3.0962689272552266) q[10];
ry(-1.5060802147382206) q[11];
rz(-1.5304363154660257) q[11];
ry(3.0323658687324997) q[12];
rz(-1.0302679372358716) q[12];
ry(-2.7997788547940154) q[13];
rz(-2.0076424332749943) q[13];
ry(1.6688628813797364) q[14];
rz(-2.475231927904017) q[14];
ry(2.086985233116488) q[15];
rz(2.092775778410897) q[15];