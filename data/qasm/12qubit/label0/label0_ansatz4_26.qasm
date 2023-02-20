OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.3929497052631655) q[0];
rz(-1.8914924661489585) q[0];
ry(-0.8640359627833233) q[1];
rz(-2.2219500996781374) q[1];
ry(3.1388732428739616) q[2];
rz(2.572948143552361) q[2];
ry(3.128543947894883) q[3];
rz(-0.0068881294580185335) q[3];
ry(-3.139331504385901) q[4];
rz(1.0276710581590685) q[4];
ry(-3.1415808951529534) q[5];
rz(2.0540861368972525) q[5];
ry(1.5717053971285138) q[6];
rz(-0.414875075066651) q[6];
ry(1.5710683155169674) q[7];
rz(-0.9727444284072532) q[7];
ry(-0.0005386081304254903) q[8];
rz(1.0361421059414027) q[8];
ry(-3.1414845436972767) q[9];
rz(-1.704606086747117) q[9];
ry(0.10138817557743973) q[10];
rz(-2.6620062265756883) q[10];
ry(-0.25148974899136745) q[11];
rz(2.24999463977189) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5373444561767655) q[0];
rz(0.8000070651935021) q[0];
ry(-2.1283492583709007) q[1];
rz(0.5443699620161457) q[1];
ry(0.021237930044738392) q[2];
rz(0.6074433527210162) q[2];
ry(-3.11386210128358) q[3];
rz(2.389906394857739) q[3];
ry(-1.5672742284942718) q[4];
rz(3.1370599619449964) q[4];
ry(-1.5702857024648635) q[5];
rz(-0.0004848123570395657) q[5];
ry(1.7968079605417955) q[6];
rz(1.1640069198048117) q[6];
ry(-2.8438077099644055) q[7];
rz(0.17586644247700708) q[7];
ry(-1.5762918416646716) q[8];
rz(0.16583870769338285) q[8];
ry(1.6271242789028078) q[9];
rz(2.72334526757342) q[9];
ry(2.4323599551393116) q[10];
rz(0.1222014174633772) q[10];
ry(2.9400441317398163) q[11];
rz(-0.8268762164331535) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.008817095624434224) q[0];
rz(-2.2289664772178015) q[0];
ry(-2.0266587110137433) q[1];
rz(-2.9551773796799687) q[1];
ry(-1.4921818436181598) q[2];
rz(-0.09448874958795185) q[2];
ry(-1.6095802054190909) q[3];
rz(-0.05932591442939462) q[3];
ry(1.8759544858902115) q[4];
rz(-1.8291101466918764) q[4];
ry(1.8770063626800322) q[5];
rz(-1.3975691518102413) q[5];
ry(-0.8255426772564914) q[6];
rz(-0.8910960239964298) q[6];
ry(-2.316718150765218) q[7];
rz(-1.9816219035048328) q[7];
ry(0.005965990118889983) q[8];
rz(1.4692408096814917) q[8];
ry(0.004275619253956563) q[9];
rz(-1.0412129533195769) q[9];
ry(-1.6888763429523428) q[10];
rz(0.5289752207979387) q[10];
ry(1.438136510894514) q[11];
rz(1.1342997148881973) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.972128241273784) q[0];
rz(-1.9707804563622764) q[0];
ry(-0.12124111039217918) q[1];
rz(1.0537359802844435) q[1];
ry(2.765502424250174) q[2];
rz(1.5304108088021928) q[2];
ry(-0.2515619352576835) q[3];
rz(2.001714671906174) q[3];
ry(-1.480832207289361) q[4];
rz(3.0035809779165863) q[4];
ry(2.2435295635543016) q[5];
rz(1.2077084701626817) q[5];
ry(-0.5210797507411576) q[6];
rz(0.4426270394978556) q[6];
ry(-2.2520459753443536) q[7];
rz(-1.963204551733541) q[7];
ry(1.1602456435659567) q[8];
rz(1.7881266469565693) q[8];
ry(1.9290620097166187) q[9];
rz(1.9785968261829998) q[9];
ry(1.5364756443885321) q[10];
rz(2.936632504403817) q[10];
ry(-0.05548439711989883) q[11];
rz(-1.2403305089865926) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.09342171847456449) q[0];
rz(0.6438695461278217) q[0];
ry(0.7199355529028261) q[1];
rz(-1.579884285144449) q[1];
ry(0.49104466939484137) q[2];
rz(-2.3272988966227643) q[2];
ry(0.007673843039065708) q[3];
rz(-2.4784727625397123) q[3];
ry(-0.018137238586101005) q[4];
rz(-2.7129386555042494) q[4];
ry(0.36472266411577897) q[5];
rz(-0.8669828833478279) q[5];
ry(-3.1380718176637967) q[6];
rz(-3.040926037012235) q[6];
ry(-3.1381125563752916) q[7];
rz(-2.6471023365150272) q[7];
ry(3.1344586583422362) q[8];
rz(1.9639207476750231) q[8];
ry(3.040113270124345) q[9];
rz(1.9268530352443882) q[9];
ry(0.32656783790894206) q[10];
rz(-0.345769898374729) q[10];
ry(-0.9459494036447866) q[11];
rz(-1.2268618209397786) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5129150239146147) q[0];
rz(2.89315518102601) q[0];
ry(1.7040380627137595) q[1];
rz(-2.801842045737398) q[1];
ry(-3.114926214012882) q[2];
rz(0.9603420872141123) q[2];
ry(0.014504548967002151) q[3];
rz(1.6876560361703796) q[3];
ry(2.099190747023604) q[4];
rz(1.877612775187431) q[4];
ry(2.493995617662842) q[5];
rz(2.770527816417761) q[5];
ry(1.8810354773875622) q[6];
rz(1.8382026616434928) q[6];
ry(1.8789783518163041) q[7];
rz(1.8308851810509887) q[7];
ry(-2.8736277455338883) q[8];
rz(-1.644546796643603) q[8];
ry(-1.4465291852883144) q[9];
rz(-1.5650727244751135) q[9];
ry(-0.4864556880143481) q[10];
rz(-2.9386391638935634) q[10];
ry(-2.7906292204451786) q[11];
rz(-2.3403137703163392) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.7342393755149725) q[0];
rz(0.6437438802709394) q[0];
ry(-3.051230274887409) q[1];
rz(-2.4842506229522723) q[1];
ry(0.514085756774671) q[2];
rz(-1.7214733444106205) q[2];
ry(-3.069690548139194) q[3];
rz(-2.6183305252231426) q[3];
ry(-2.777270667399254) q[4];
rz(-1.8796100573795838) q[4];
ry(-2.627235785690642) q[5];
rz(-0.8415598455943424) q[5];
ry(0.09666231055407781) q[6];
rz(-2.0732742828722994) q[6];
ry(0.0966860063481455) q[7];
rz(-2.1258619167770876) q[7];
ry(3.0790926164964305) q[8];
rz(-1.700425867016599) q[8];
ry(0.07119910762295056) q[9];
rz(1.524958187818805) q[9];
ry(3.0829689315994955) q[10];
rz(-2.569733602201404) q[10];
ry(-2.9088662185576797) q[11];
rz(-1.5892918054101797) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.6267428632747905) q[0];
rz(-2.385275758071349) q[0];
ry(0.45711051251645135) q[1];
rz(0.7184424123606694) q[1];
ry(-3.107713226149941) q[2];
rz(-0.29248003264866096) q[2];
ry(-3.0797156903465517) q[3];
rz(-1.2335195903101541) q[3];
ry(-2.5972151796056) q[4];
rz(2.697414476255884) q[4];
ry(1.2672427477007648) q[5];
rz(-2.019406443466372) q[5];
ry(-0.033077880879393895) q[6];
rz(-1.778355829269609) q[6];
ry(-0.04029709494283471) q[7];
rz(1.4169690880635013) q[7];
ry(2.228858834495507) q[8];
rz(0.8051438725574409) q[8];
ry(0.38313259242210335) q[9];
rz(-0.5322238136103544) q[9];
ry(1.210713765803506) q[10];
rz(-2.1687681599472532) q[10];
ry(2.123829148144554) q[11];
rz(0.25637902270047397) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.4154756806434623) q[0];
rz(1.6961290275561502) q[0];
ry(-0.21720633659386568) q[1];
rz(3.0129905652295044) q[1];
ry(0.7724783367483905) q[2];
rz(1.4417843781410244) q[2];
ry(0.9901009655779902) q[3];
rz(-2.7200084523562373) q[3];
ry(-3.138655796972962) q[4];
rz(-1.79306000215748) q[4];
ry(-2.207631255232644) q[5];
rz(-2.3845024951033476) q[5];
ry(1.493679130342148) q[6];
rz(-2.855938671813428) q[6];
ry(1.6371090797583614) q[7];
rz(-2.141106859352787) q[7];
ry(-3.107048865230073) q[8];
rz(-0.9163973594863702) q[8];
ry(-0.014467964412914544) q[9];
rz(2.38467779299958) q[9];
ry(1.3305652232583332) q[10];
rz(0.25087354515111904) q[10];
ry(-0.8171866256984391) q[11];
rz(-2.8496616993099537) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.8365648709829863) q[0];
rz(-3.104095889512121) q[0];
ry(-0.8325942264181938) q[1];
rz(2.9619118493318974) q[1];
ry(-0.004520831451481187) q[2];
rz(-1.0490427750816647) q[2];
ry(0.06325732605381518) q[3];
rz(-0.5639695537304245) q[3];
ry(-3.070292979055702) q[4];
rz(0.6408861815369303) q[4];
ry(-0.03177252612777348) q[5];
rz(1.6071113807152897) q[5];
ry(3.1214370530468303) q[6];
rz(-0.11129608554201777) q[6];
ry(3.1380777182863397) q[7];
rz(1.4287833814307345) q[7];
ry(-1.2742818147215083) q[8];
rz(3.081148508127508) q[8];
ry(-1.5647322826155994) q[9];
rz(-1.7127595771769097) q[9];
ry(-2.361072135630528) q[10];
rz(1.9996641105364032) q[10];
ry(0.1190549901568998) q[11];
rz(-2.9063671134891194) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.178314886192802) q[0];
rz(-1.0640874346895783) q[0];
ry(-2.0260668893910507) q[1];
rz(2.340193509270014) q[1];
ry(3.0108831141360004) q[2];
rz(0.37016368650668663) q[2];
ry(-2.801718414904212) q[3];
rz(-1.0938162648750442) q[3];
ry(2.186588828725643) q[4];
rz(-0.9097182391507806) q[4];
ry(-1.9166464505243044) q[5];
rz(1.5363703958695076) q[5];
ry(-2.549371769460446) q[6];
rz(2.7625560346188878) q[6];
ry(-0.5947941218776647) q[7];
rz(-0.4119111272610061) q[7];
ry(1.5329498149237022) q[8];
rz(2.259934540132269) q[8];
ry(1.695794908312065) q[9];
rz(1.3662056798852866) q[9];
ry(-2.1503409155912707) q[10];
rz(-1.530998002793961) q[10];
ry(2.630583902218781) q[11];
rz(-3.0488437397824435) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.7457527672889771) q[0];
rz(2.5137798425479327) q[0];
ry(2.582600472245529) q[1];
rz(-1.6685618155213942) q[1];
ry(1.694038137261721) q[2];
rz(-1.1721297123011876) q[2];
ry(1.383085875398164) q[3];
rz(-1.8985574181756943) q[3];
ry(-0.1868088172242027) q[4];
rz(-0.45678181741670615) q[4];
ry(-2.053736310346642) q[5];
rz(-2.232509209021297) q[5];
ry(-2.7019540166680835) q[6];
rz(1.6407864706889355) q[6];
ry(0.46038417961917444) q[7];
rz(-1.5545322970594038) q[7];
ry(0.012086712731341542) q[8];
rz(0.29227664862725344) q[8];
ry(3.0806421139396623) q[9];
rz(3.076873995101772) q[9];
ry(2.419517792752572) q[10];
rz(2.6836990443136006) q[10];
ry(-2.4191079664633053) q[11];
rz(0.983451738036619) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.6373833244506422) q[0];
rz(-1.1705539166807073) q[0];
ry(-2.540318106610587) q[1];
rz(-2.08336933973838) q[1];
ry(1.6366909842378796) q[2];
rz(1.7243163130664003) q[2];
ry(1.5211096311876675) q[3];
rz(1.7108940927609702) q[3];
ry(3.0619251178624394) q[4];
rz(-0.6300302982241351) q[4];
ry(-0.2515403275619219) q[5];
rz(0.43959632117776337) q[5];
ry(-1.2194741986769226) q[6];
rz(-2.6654997977865507) q[6];
ry(-2.1601875095232193) q[7];
rz(1.9693754824318708) q[7];
ry(0.04207908780637564) q[8];
rz(1.4460866070721197) q[8];
ry(3.086908861912941) q[9];
rz(-0.8096643266897656) q[9];
ry(-1.7627475732881979) q[10];
rz(1.9358066917782568) q[10];
ry(-0.7091841046268216) q[11];
rz(-1.49719348719981) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.786967334333466) q[0];
rz(0.7808659703505275) q[0];
ry(-0.8690277501233826) q[1];
rz(0.3664747961880742) q[1];
ry(1.9321411611167965) q[2];
rz(1.3041414193051408) q[2];
ry(-1.865373650584102) q[3];
rz(2.257643875112072) q[3];
ry(0.013191433916926691) q[4];
rz(-1.8226562834771176) q[4];
ry(-1.4917986690357212) q[5];
rz(-1.546210936986725) q[5];
ry(-0.03157186999558981) q[6];
rz(3.016210644426057) q[6];
ry(1.503205736270232) q[7];
rz(1.403972799103725) q[7];
ry(0.01081377579444709) q[8];
rz(1.4159105675130421) q[8];
ry(0.010507377325139624) q[9];
rz(-0.3268996765939648) q[9];
ry(-2.991732612310204) q[10];
rz(-2.5407020847267305) q[10];
ry(-1.9106541704044337) q[11];
rz(-0.7297194163149263) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.421185684143488) q[0];
rz(-2.514839966657506) q[0];
ry(2.526543699058207) q[1];
rz(0.5243724253140281) q[1];
ry(-0.014858659483557557) q[2];
rz(-0.9474869976968253) q[2];
ry(0.01869896683089363) q[3];
rz(-2.5328831982911244) q[3];
ry(1.5829895911042833) q[4];
rz(3.108399916283226) q[4];
ry(1.562075745781061) q[5];
rz(-0.06113897386498212) q[5];
ry(1.5372553934514028) q[6];
rz(-0.007656441774724272) q[6];
ry(-1.5917769901228231) q[7];
rz(0.10016567960764612) q[7];
ry(3.1364203788918426) q[8];
rz(2.2479013493584703) q[8];
ry(-3.1383886276128004) q[9];
rz(-2.855936759831845) q[9];
ry(0.2999770263177849) q[10];
rz(0.46373856233284816) q[10];
ry(1.3248579450509759) q[11];
rz(-0.42747827358924356) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.1789818074501826) q[0];
rz(0.49468400418853775) q[0];
ry(1.5861767108024116) q[1];
rz(-1.3785707314634241) q[1];
ry(-3.1337516367313683) q[2];
rz(-1.5686308998615468) q[2];
ry(-0.004855540503518632) q[3];
rz(-0.9307586311759514) q[3];
ry(-1.5713940294852238) q[4];
rz(2.6817727603665746) q[4];
ry(1.5750968012907052) q[5];
rz(-1.5763234254478071) q[5];
ry(1.5566391328785696) q[6];
rz(-1.5885552060990644) q[6];
ry(0.0027388189906457196) q[7];
rz(2.244009194556077) q[7];
ry(-2.4374161427841368) q[8];
rz(1.6278621460236362) q[8];
ry(0.6889156548708091) q[9];
rz(-1.4038910945421392) q[9];
ry(0.19213376879668906) q[10];
rz(-2.7373134750008945) q[10];
ry(-1.6503585809775998) q[11];
rz(-2.2778772650277945) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.3326866224923946) q[0];
rz(-0.7552167298339556) q[0];
ry(1.904766164809411) q[1];
rz(-1.8646893779879623) q[1];
ry(0.00838496999418492) q[2];
rz(2.1488322115260257) q[2];
ry(0.00047568095179606473) q[3];
rz(2.728069259798669) q[3];
ry(3.1374649698015467) q[4];
rz(-2.0123102193415723) q[4];
ry(3.1359052187716285) q[5];
rz(-3.12580484737268) q[5];
ry(1.5668246726880937) q[6];
rz(1.6367792756520008) q[6];
ry(-3.1333443254312066) q[7];
rz(2.340422843707519) q[7];
ry(-0.0002160370539880293) q[8];
rz(-1.5769425775442076) q[8];
ry(3.1408231638513655) q[9];
rz(-1.454611509819865) q[9];
ry(-2.9896230303147884) q[10];
rz(2.0343431857153136) q[10];
ry(-0.4913366955091307) q[11];
rz(-1.355653526865696) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.765016237986851) q[0];
rz(2.8944598582889904) q[0];
ry(-1.9388861643736803) q[1];
rz(0.5289878140276388) q[1];
ry(-0.029302528842213228) q[2];
rz(2.525367047695415) q[2];
ry(0.10290542091606003) q[3];
rz(-1.3001857168964008) q[3];
ry(-1.8764870851887765) q[4];
rz(-1.3680738435134046) q[4];
ry(1.6025474226467924) q[5];
rz(1.299052890620123) q[5];
ry(-0.012449850690535565) q[6];
rz(2.944630551481042) q[6];
ry(1.5820255003743693) q[7];
rz(0.086730972531333) q[7];
ry(-1.7816026264832052) q[8];
rz(-1.4058404203237567) q[8];
ry(1.7699224823100035) q[9];
rz(3.0166961896971167) q[9];
ry(2.657308121379508) q[10];
rz(1.1869917521375504) q[10];
ry(1.2158881069595804) q[11];
rz(-2.6172631063753213) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.581837015031715) q[0];
rz(0.12695717226492942) q[0];
ry(2.4350054241256465) q[1];
rz(-1.0445156256397707) q[1];
ry(-0.0065716441697372795) q[2];
rz(-1.2342209156747588) q[2];
ry(-3.1357974507431106) q[3];
rz(-1.4147562241911635) q[3];
ry(-1.699735515869948) q[4];
rz(1.1321304054695152) q[4];
ry(1.6092098121213505) q[5];
rz(-1.5079621385562956) q[5];
ry(0.2435737831390413) q[6];
rz(1.4181109718961766) q[6];
ry(-1.4480883813325303) q[7];
rz(-2.2515742468006374) q[7];
ry(0.10810686337830425) q[8];
rz(2.967931470030836) q[8];
ry(0.7951753691963398) q[9];
rz(1.7407320155868207) q[9];
ry(0.025127932406827647) q[10];
rz(3.0520933410328346) q[10];
ry(1.5091987860552685) q[11];
rz(-0.3883221327707996) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.3701328874604286) q[0];
rz(-2.376166038989724) q[0];
ry(-1.951185431725792) q[1];
rz(-2.220041875066924) q[1];
ry(-1.6048274861743608) q[2];
rz(-2.7890652253551487) q[2];
ry(1.5952802447172185) q[3];
rz(3.131817238636663) q[3];
ry(0.007369805205279966) q[4];
rz(-1.2493995905629447) q[4];
ry(-0.040699059114144544) q[5];
rz(-2.089587871205576) q[5];
ry(0.0010694224815106684) q[6];
rz(2.05602643355786) q[6];
ry(-3.1388438989024805) q[7];
rz(-1.0888425272253375) q[7];
ry(-1.6985016703477145) q[8];
rz(0.1298043999393652) q[8];
ry(-1.4467682215977549) q[9];
rz(2.3177569749515143) q[9];
ry(-2.617507377521991) q[10];
rz(-1.6783430175859566) q[10];
ry(-2.1842398177193996) q[11];
rz(2.5937045027441643) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.8239274285856077) q[0];
rz(-0.39592273631835695) q[0];
ry(0.3506177057982125) q[1];
rz(-1.5111822031447717) q[1];
ry(0.1241613251414635) q[2];
rz(1.2202825405483084) q[2];
ry(-1.9453511309687075) q[3];
rz(-1.6158172366946908) q[3];
ry(-1.7585628670445834) q[4];
rz(2.736206041702307) q[4];
ry(0.38623897371712346) q[5];
rz(1.310466306704355) q[5];
ry(3.134576103267738) q[6];
rz(-2.0994216795669525) q[6];
ry(0.2606123858920064) q[7];
rz(-0.1205205607676012) q[7];
ry(-1.1783360537962606) q[8];
rz(-2.1007619255986487) q[8];
ry(1.9148782401180027) q[9];
rz(-1.1163292108104939) q[9];
ry(-0.571131315977488) q[10];
rz(-1.8282685007012784) q[10];
ry(1.654802152805276) q[11];
rz(-2.8001548108982) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.4150808964252637) q[0];
rz(-1.7193223692788662) q[0];
ry(2.918444927744028) q[1];
rz(-3.0345473058658614) q[1];
ry(1.571628748171169) q[2];
rz(1.4202450859677793) q[2];
ry(1.5707754481237963) q[3];
rz(-1.5168803918679377) q[3];
ry(-1.3994948944112524) q[4];
rz(-1.4167270096318032) q[4];
ry(-0.17940659011969642) q[5];
rz(-0.47577635058213724) q[5];
ry(-3.141151707436277) q[6];
rz(-0.37583892153027953) q[6];
ry(3.1407618033993243) q[7];
rz(-1.4351043811486808) q[7];
ry(-3.1120339456102775) q[8];
rz(-0.8735907584175999) q[8];
ry(-0.05655062564260636) q[9];
rz(-2.109887280865714) q[9];
ry(2.006619369729293) q[10];
rz(-1.3615030459934578) q[10];
ry(-2.3043738905476436) q[11];
rz(-0.31616449737313435) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.2947534628746604) q[0];
rz(2.7289280582501254) q[0];
ry(-0.46561041378707374) q[1];
rz(-1.6285988675467662) q[1];
ry(-0.0007708246306163957) q[2];
rz(-1.1396328578780155) q[2];
ry(0.0008163987609371242) q[3];
rz(-2.234897904539821) q[3];
ry(1.5171941505289446) q[4];
rz(0.935342842602604) q[4];
ry(-1.5695032131546034) q[5];
rz(0.16659885662839777) q[5];
ry(3.1414307654827467) q[6];
rz(0.9393316948874847) q[6];
ry(-0.00019474110078183108) q[7];
rz(2.879598743516635) q[7];
ry(-0.15843993690854585) q[8];
rz(2.0017358126357845) q[8];
ry(0.4704632963244739) q[9];
rz(0.20985033151548355) q[9];
ry(0.9489941729224789) q[10];
rz(1.7458618353586481) q[10];
ry(0.746083066046804) q[11];
rz(0.9655932330587264) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.1313911886006554) q[0];
rz(-1.952109998742676) q[0];
ry(1.08920239775562) q[1];
rz(-0.7444144587509481) q[1];
ry(-3.1413097901296614) q[2];
rz(1.105750106746409) q[2];
ry(-0.001485912658343451) q[3];
rz(-2.2925098910173873) q[3];
ry(1.503799644767911) q[4];
rz(-0.03383975997413379) q[4];
ry(0.8118110906001306) q[5];
rz(-1.862662063275223) q[5];
ry(3.14127277771904) q[6];
rz(-0.5693716027153515) q[6];
ry(3.141129401859823) q[7];
rz(-2.215218623474901) q[7];
ry(0.14036000791978953) q[8];
rz(-0.4010169613095564) q[8];
ry(3.0713904270417007) q[9];
rz(2.8927161965281774) q[9];
ry(2.1200798423010276) q[10];
rz(2.68472365377459) q[10];
ry(2.6572336938632115) q[11];
rz(-0.583228051160356) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.907428648253345) q[0];
rz(2.0651213179679577) q[0];
ry(1.4235748205499874) q[1];
rz(0.24361170155983508) q[1];
ry(-3.1366628353763373) q[2];
rz(-2.2218050002454333) q[2];
ry(-3.1386408233430476) q[3];
rz(-1.1867699638024987) q[3];
ry(1.6802878345708825) q[4];
rz(1.4899857700776975) q[4];
ry(-1.595764437496197) q[5];
rz(1.2930514179198216) q[5];
ry(1.6923394003935943) q[6];
rz(-0.9297335068608232) q[6];
ry(-0.015684463426900875) q[7];
rz(1.774750662278226) q[7];
ry(0.7928669389680782) q[8];
rz(-1.0155503747409398) q[8];
ry(-1.3826645409881362) q[9];
rz(-0.3659374167144867) q[9];
ry(-2.6986920188782184) q[10];
rz(-0.16062381463574135) q[10];
ry(2.457793432563693) q[11];
rz(2.2556602272195887) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.0317319997011962) q[0];
rz(0.7867475195726332) q[0];
ry(1.6055016440779484) q[1];
rz(1.668781346462663) q[1];
ry(-3.140565082088777) q[2];
rz(2.0439623872994837) q[2];
ry(-3.1410669805465283) q[3];
rz(-1.920859584767755) q[3];
ry(3.141183204955711) q[4];
rz(-2.959911345549722) q[4];
ry(3.1392988649364786) q[5];
rz(1.5307158752710968) q[5];
ry(3.1413258892829155) q[6];
rz(2.2081863317775405) q[6];
ry(-0.0004936623159589186) q[7];
rz(0.8983722589629365) q[7];
ry(-0.001310398180376815) q[8];
rz(0.2693335956377933) q[8];
ry(3.140814233852474) q[9];
rz(1.5822099641644927) q[9];
ry(-2.884561577326985) q[10];
rz(2.36848213702781) q[10];
ry(-1.554965660441777) q[11];
rz(1.571547718648773) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.8825222273046237) q[0];
rz(0.30868358586846695) q[0];
ry(-2.975729511199486) q[1];
rz(-0.020353163226983995) q[1];
ry(3.1376060506856085) q[2];
rz(-1.710369901498705) q[2];
ry(-3.1378210814734566) q[3];
rz(-2.198709632090117) q[3];
ry(3.0561478137118105) q[4];
rz(-0.1278614014376913) q[4];
ry(-2.9263756409469557) q[5];
rz(-2.700622967296163) q[5];
ry(1.4480989908307844) q[6];
rz(2.5574860252160487) q[6];
ry(3.1324028768860392) q[7];
rz(0.07620637681211213) q[7];
ry(-1.5270744065773565) q[8];
rz(-2.9227336245620035) q[8];
ry(0.912890724765381) q[9];
rz(-0.3656538670118363) q[9];
ry(-2.449832344542163) q[10];
rz(2.657685445956382) q[10];
ry(1.1841478594269808) q[11];
rz(-3.0054355572811224) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.927301742270547) q[0];
rz(2.6702309764970225) q[0];
ry(0.9368591923082061) q[1];
rz(0.11672932787424305) q[1];
ry(-1.315894895817891) q[2];
rz(-0.4422933703279418) q[2];
ry(1.5288183010721987) q[3];
rz(0.15868698291709293) q[3];
ry(1.677302554457084) q[4];
rz(-0.6127012464295182) q[4];
ry(1.08109098861299) q[5];
rz(-0.2065580552948454) q[5];
ry(3.10789636200468) q[6];
rz(-2.0411119229521346) q[6];
ry(-2.830606008394561) q[7];
rz(3.111189922476546) q[7];
ry(-1.5519958414289992) q[8];
rz(-0.05335632589873958) q[8];
ry(-1.5625680490094294) q[9];
rz(3.141489316040269) q[9];
ry(-0.8899039301246923) q[10];
rz(3.1331047081693146) q[10];
ry(-1.7871785687228812) q[11];
rz(0.9715578565907628) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.2956246727269314) q[0];
rz(-3.1341736513401797) q[0];
ry(0.11561421016374766) q[1];
rz(-1.3063741736217738) q[1];
ry(-3.139496564054422) q[2];
rz(2.4897508731496014) q[2];
ry(0.0010619860296606162) q[3];
rz(-0.08400571928792044) q[3];
ry(-3.1413762792092963) q[4];
rz(2.9514549053983914) q[4];
ry(-1.354941167175383e-05) q[5];
rz(2.350001592368927) q[5];
ry(3.1414770973495747) q[6];
rz(2.6833145763349213) q[6];
ry(-0.0004392957715575463) q[7];
rz(-1.5203809289038934) q[7];
ry(-1.7086317996360139) q[8];
rz(-1.8586565350895272) q[8];
ry(-1.483147610233652) q[9];
rz(1.6732204771812444) q[9];
ry(2.8702342000354877) q[10];
rz(-0.30170876012414194) q[10];
ry(-0.7908440373165398) q[11];
rz(-0.8860559087151215) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.1012191782233534) q[0];
rz(1.6651054335726416) q[0];
ry(3.1308821541048504) q[1];
rz(0.5708494138267662) q[1];
ry(-1.5514548453031631) q[2];
rz(-1.064942784414332) q[2];
ry(-1.6504686805377227) q[3];
rz(-2.90327586089329) q[3];
ry(2.9869940843373466) q[4];
rz(-2.6419868389824104) q[4];
ry(2.5462496409441844) q[5];
rz(-2.0753780205132975) q[5];
ry(-1.5866925774174359) q[6];
rz(-3.0808525445021977) q[6];
ry(-1.5451708811848563) q[7];
rz(-1.8837725043565143) q[7];
ry(-0.048955239505131636) q[8];
rz(2.868620664312397) q[8];
ry(2.0318220606322637) q[9];
rz(-1.6631551533769553) q[9];
ry(0.8375403330269359) q[10];
rz(0.4300958340403005) q[10];
ry(1.7634557211876958) q[11];
rz(-0.08015871387394578) q[11];