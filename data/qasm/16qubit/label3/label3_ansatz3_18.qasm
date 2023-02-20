OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5470874269791732) q[0];
rz(-1.583662717282273) q[0];
ry(0.0005787357173590024) q[1];
rz(-1.9352107558914469) q[1];
ry(-2.720955684821676) q[2];
rz(-2.9346247837240584) q[2];
ry(-0.0005548894782387848) q[3];
rz(-2.5997709026195697) q[3];
ry(1.0272511390289987) q[4];
rz(-0.3928399047981483) q[4];
ry(3.1308635479548492) q[5];
rz(1.513046623738072) q[5];
ry(0.00019784577384193163) q[6];
rz(1.290381786874775) q[6];
ry(1.5418011608440247) q[7];
rz(1.5298341637824704) q[7];
ry(-3.1228605673893948) q[8];
rz(-0.20753860547559652) q[8];
ry(-0.00808454845979013) q[9];
rz(-2.8993794472602303) q[9];
ry(-2.3705732218758886) q[10];
rz(2.968962018035434) q[10];
ry(0.6659922574174842) q[11];
rz(-1.5472317909559488) q[11];
ry(3.1407886968503655) q[12];
rz(2.4333342797241433) q[12];
ry(-0.044791664793672054) q[13];
rz(-2.8016594870280866) q[13];
ry(-0.00508804913375549) q[14];
rz(1.468625405672075) q[14];
ry(0.37423851128601626) q[15];
rz(-1.9533411413263297) q[15];
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
ry(2.1972563382995296) q[0];
rz(-0.08674557872686962) q[0];
ry(-1.3559819685043135) q[1];
rz(-0.15191411694398393) q[1];
ry(1.8604511598327285) q[2];
rz(-3.1342722572620363) q[2];
ry(-3.1396519114200885) q[3];
rz(-0.25137351933980084) q[3];
ry(-0.9595024938338348) q[4];
rz(2.751643436634418) q[4];
ry(-0.04299969699570916) q[5];
rz(1.1444699406189311) q[5];
ry(0.1021948226273525) q[6];
rz(-2.2044144740396905) q[6];
ry(-1.333141439692488) q[7];
rz(-3.1080989874883134) q[7];
ry(-0.04125891535656301) q[8];
rz(-1.0988430599862278) q[8];
ry(1.5645022503696895) q[9];
rz(2.3792001794020905) q[9];
ry(-2.503628077844708) q[10];
rz(-1.3952332853126448) q[10];
ry(-1.8389349457986308) q[11];
rz(-2.6049349542790297) q[11];
ry(-0.4358494873629715) q[12];
rz(1.3086386529585647) q[12];
ry(0.07333655228982126) q[13];
rz(2.968680425161824) q[13];
ry(-0.01373881147499123) q[14];
rz(2.9174368457077304) q[14];
ry(2.449357788833805) q[15];
rz(-0.47807244932746296) q[15];
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
ry(-2.719806167300718) q[0];
rz(-0.7171157268349511) q[0];
ry(-0.5112459749573457) q[1];
rz(-2.571148350767883) q[1];
ry(-0.8800459934132822) q[2];
rz(1.0147318985349019) q[2];
ry(0.002355266175909421) q[3];
rz(3.046549211721041) q[3];
ry(-0.00764359743460779) q[4];
rz(1.7183588672591013) q[4];
ry(0.7531231904779974) q[5];
rz(-2.720697441646985) q[5];
ry(3.1394445147163945) q[6];
rz(-0.4980687702884268) q[6];
ry(-2.407826389594176) q[7];
rz(2.535918927318021) q[7];
ry(-1.549839244634522) q[8];
rz(2.076226885775891) q[8];
ry(2.8443754479724768) q[9];
rz(-1.4978243094255983) q[9];
ry(1.5992005121371422) q[10];
rz(0.001580068113142798) q[10];
ry(0.03544226278504237) q[11];
rz(3.080124291368403) q[11];
ry(-0.5133221815971032) q[12];
rz(-2.494424504570969) q[12];
ry(-1.4417457010499302) q[13];
rz(0.07218134767463569) q[13];
ry(-0.01242699845023612) q[14];
rz(-1.5772906525694832) q[14];
ry(2.3045806975843384) q[15];
rz(-2.1562097387722012) q[15];
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
ry(1.683292263147376) q[0];
rz(0.13668285519784748) q[0];
ry(0.16988644243655351) q[1];
rz(-1.2626732184916993) q[1];
ry(-0.5730693169833403) q[2];
rz(0.569491309427849) q[2];
ry(-0.00034644727464751984) q[3];
rz(1.6115231226128) q[3];
ry(-0.624703999554632) q[4];
rz(-0.8185316408914075) q[4];
ry(1.9950783279375202) q[5];
rz(-2.2634669750446044) q[5];
ry(3.1265669911292897) q[6];
rz(-1.4219112921166843) q[6];
ry(-3.1283422986737057) q[7];
rz(-3.094615302902686) q[7];
ry(0.00948130728443287) q[8];
rz(2.5521379241178916) q[8];
ry(-1.59208242367884) q[9];
rz(0.5759699650105716) q[9];
ry(0.004690013244416004) q[10];
rz(2.2731184550571406) q[10];
ry(-2.9764870450259404) q[11];
rz(-2.931774109051039) q[11];
ry(2.857649002535149) q[12];
rz(1.3170808927626956) q[12];
ry(3.1184814596175863) q[13];
rz(2.3082527446001855) q[13];
ry(-1.485899681369343) q[14];
rz(1.637471675730633) q[14];
ry(-2.313328943251056) q[15];
rz(-2.4179857743319033) q[15];
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
ry(2.9416394176115253) q[0];
rz(-2.369223240401153) q[0];
ry(0.9428666985315294) q[1];
rz(-1.964442835095678) q[1];
ry(1.9826896751471832) q[2];
rz(-0.0185562420345029) q[2];
ry(-3.1264476252181375) q[3];
rz(-1.826584248810092) q[3];
ry(-3.1404020578107685) q[4];
rz(-0.014697364444524295) q[4];
ry(0.8120004706640939) q[5];
rz(0.5233055315334894) q[5];
ry(1.7980917969298174) q[6];
rz(0.0025475730910260778) q[6];
ry(-2.7036612884227553) q[7];
rz(-1.4759802385024705) q[7];
ry(-1.5545905259852537) q[8];
rz(1.8027889324003836) q[8];
ry(-0.12414997534587567) q[9];
rz(2.4459491773756183) q[9];
ry(0.9151471434395038) q[10];
rz(0.934371109644178) q[10];
ry(1.2509111399131285) q[11];
rz(0.4982062111283394) q[11];
ry(-2.9093296088562552) q[12];
rz(3.1208126548123176) q[12];
ry(-2.2944720922485033) q[13];
rz(1.6531952626309865) q[13];
ry(0.08697214878903865) q[14];
rz(-2.186601681290914) q[14];
ry(0.04831606808493128) q[15];
rz(0.9028647849547555) q[15];
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
ry(0.6475044098555598) q[0];
rz(1.401533530574697) q[0];
ry(0.8644062685869978) q[1];
rz(1.8437012843931875) q[1];
ry(-1.2523221603210697) q[2];
rz(2.792078327503625) q[2];
ry(-3.1388626257612633) q[3];
rz(-1.251699451374943) q[3];
ry(-0.0006280049112231468) q[4];
rz(0.22379278635225217) q[4];
ry(0.017890507546924894) q[5];
rz(1.9532037039874952) q[5];
ry(1.5038348204957472) q[6];
rz(-3.141252935684371) q[6];
ry(0.10299739763226405) q[7];
rz(-1.895771237612954) q[7];
ry(-3.1407246313748436) q[8];
rz(2.0480049351556286) q[8];
ry(-0.021673667585759837) q[9];
rz(0.5698096154745711) q[9];
ry(-0.004545007984913135) q[10];
rz(-1.4515154692972205) q[10];
ry(-0.03905388225375717) q[11];
rz(1.9419082138102688) q[11];
ry(-1.4796914473501186) q[12];
rz(-2.8545081959763943) q[12];
ry(3.1151645087364974) q[13];
rz(-2.042940312174064) q[13];
ry(3.0333847486618035) q[14];
rz(-1.8025310642040686) q[14];
ry(-1.441982337526233) q[15];
rz(2.1830703130650524) q[15];
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
ry(1.3623626662038073) q[0];
rz(-1.754934242725814) q[0];
ry(3.0649310289995997) q[1];
rz(2.1555488382937646) q[1];
ry(-1.8721661936643008) q[2];
rz(1.5502235120692904) q[2];
ry(-3.1365557889927786) q[3];
rz(-2.794281518751201) q[3];
ry(1.5752915206677534) q[4];
rz(-3.056890963527432) q[4];
ry(1.5828786640577794) q[5];
rz(-1.3714702997840842) q[5];
ry(1.345324442412399) q[6];
rz(1.4390823505980763) q[6];
ry(0.02626388404569369) q[7];
rz(-1.2356520705324905) q[7];
ry(3.130012662121813) q[8];
rz(-2.8828658460862577) q[8];
ry(0.14343092721964834) q[9];
rz(-1.8995922144709174) q[9];
ry(0.32754812409576317) q[10];
rz(1.7663373110692548) q[10];
ry(0.4216875029575967) q[11];
rz(-2.7091743517652467) q[11];
ry(1.489034655538202) q[12];
rz(-0.06724387005982813) q[12];
ry(-1.6097630299658916) q[13];
rz(-1.5903428447559325) q[13];
ry(-0.03835966608482) q[14];
rz(1.2582097725604289) q[14];
ry(0.027064752030382982) q[15];
rz(-0.033247802649131096) q[15];
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
ry(1.1300537242507032) q[0];
rz(-1.0703644815390827) q[0];
ry(0.9033515459262385) q[1];
rz(-2.063378697776038) q[1];
ry(-2.664923392664624) q[2];
rz(1.1741330128288976) q[2];
ry(3.1400738488672753) q[3];
rz(-2.234526296915594) q[3];
ry(-0.04497662421851434) q[4];
rz(2.5204524026977135) q[4];
ry(-1.5628309674474448) q[5];
rz(0.5195376111093009) q[5];
ry(3.1170471665265986) q[6];
rz(-0.33215578738613516) q[6];
ry(1.5806419466802488) q[7];
rz(1.5061538819070268) q[7];
ry(1.1492544658705528) q[8];
rz(0.044327915921537946) q[8];
ry(1.8977262840618323) q[9];
rz(0.5276244918831434) q[9];
ry(-0.0036094976225857778) q[10];
rz(-1.6648418776834992) q[10];
ry(0.34578537863800557) q[11];
rz(-3.102472900456379) q[11];
ry(1.4074987896282982) q[12];
rz(-3.1399263978052048) q[12];
ry(-1.5731763773948648) q[13];
rz(-1.6279808806628022) q[13];
ry(2.238769374398017) q[14];
rz(3.103807035586269) q[14];
ry(1.714786547344727) q[15];
rz(-2.8595508655823165) q[15];
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
ry(-2.228826383739804) q[0];
rz(-2.3126265845927825) q[0];
ry(-1.073537383951326) q[1];
rz(1.4563552192092726) q[1];
ry(1.9546734305896667) q[2];
rz(-0.7945192290629653) q[2];
ry(0.0006158905396437594) q[3];
rz(-0.19300995629434878) q[3];
ry(0.373456230558159) q[4];
rz(0.9983148025951731) q[4];
ry(1.6694385033654258) q[5];
rz(-1.5517238779148175) q[5];
ry(-0.0005077369199497639) q[6];
rz(-1.3829543398476165) q[6];
ry(1.5823288271688332) q[7];
rz(3.105481539150533) q[7];
ry(-0.0031946823828680835) q[8];
rz(-2.9106715658190914) q[8];
ry(3.135784707762987) q[9];
rz(-2.73286477097844) q[9];
ry(-0.3870318519821368) q[10];
rz(-2.2237090729765066) q[10];
ry(-0.000499064662528248) q[11];
rz(1.4648396283969687) q[11];
ry(-1.6103804228109444) q[12];
rz(-3.1369184116898707) q[12];
ry(-1.6379375790869308) q[13];
rz(-0.19577029285284095) q[13];
ry(1.5710080715093362) q[14];
rz(0.0588370744434311) q[14];
ry(-1.620945781108168) q[15];
rz(-1.0313135492593903) q[15];
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
ry(0.9673504063328179) q[0];
rz(2.0480986502596537) q[0];
ry(-0.3425571550036877) q[1];
rz(-0.18591565123003842) q[1];
ry(-1.5516989826912848) q[2];
rz(-3.13321196223337) q[2];
ry(0.0030244351283199578) q[3];
rz(0.0573566090644925) q[3];
ry(3.137452984532244) q[4];
rz(0.8157726882203473) q[4];
ry(-3.1412576526144145) q[5];
rz(-2.3433498427826636) q[5];
ry(-2.8951200824340333) q[6];
rz(0.9333478039824152) q[6];
ry(-0.025592749335167437) q[7];
rz(0.03659240313715576) q[7];
ry(-2.649352679559746) q[8];
rz(1.569393360864621) q[8];
ry(3.056908711175727) q[9];
rz(2.9479169769650158) q[9];
ry(0.22940580801078744) q[10];
rz(-1.262266503076335) q[10];
ry(0.47149748899692434) q[11];
rz(-0.6201810927184684) q[11];
ry(1.6224457242693908) q[12];
rz(-3.1186571778338714) q[12];
ry(1.09099795693942) q[13];
rz(-0.20390665146869122) q[13];
ry(2.967007106878747) q[14];
rz(-3.0058643151504714) q[14];
ry(0.0874382755121571) q[15];
rz(-0.5482901654681648) q[15];
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
ry(1.4697949961010253) q[0];
rz(-1.99767352339746) q[0];
ry(-1.5253642221356296) q[1];
rz(0.1567709409101058) q[1];
ry(-1.546424716453961) q[2];
rz(-0.1327019610431213) q[2];
ry(1.5716082852278368) q[3];
rz(-0.5218389880473266) q[3];
ry(-1.5215114190840346) q[4];
rz(0.05515902857458393) q[4];
ry(2.4873465707446982) q[5];
rz(0.9979618168614931) q[5];
ry(-0.002646187054035383) q[6];
rz(-2.3098391703490373) q[6];
ry(-1.5425843693966768) q[7];
rz(0.38935713037401065) q[7];
ry(-0.00032668906720445534) q[8];
rz(-1.582043117502514) q[8];
ry(-3.134693466372574) q[9];
rz(2.962013685920959) q[9];
ry(-0.0009354098132856699) q[10];
rz(-1.060671695464581) q[10];
ry(-0.00016890777856515512) q[11];
rz(0.605543585327653) q[11];
ry(3.1221489784991907) q[12];
rz(0.05292820402963961) q[12];
ry(2.4555137470115103) q[13];
rz(-1.6469020808747796) q[13];
ry(2.978287882745997) q[14];
rz(-3.1246676282629102) q[14];
ry(-1.7296040804889012) q[15];
rz(1.5453363336910717) q[15];
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
ry(-1.2374372983865558) q[0];
rz(1.112059072494799) q[0];
ry(0.02977172475417511) q[1];
rz(2.752899050970531) q[1];
ry(1.5780027089893884) q[2];
rz(-0.8892067250215012) q[2];
ry(0.010404636323010348) q[3];
rz(0.5279851373312806) q[3];
ry(0.00011181359498476509) q[4];
rz(0.24074265121449212) q[4];
ry(-3.137536711448379) q[5];
rz(1.9864610164440029) q[5];
ry(-0.002600093861051689) q[6];
rz(-2.4777735712673477) q[6];
ry(-0.045176867993224123) q[7];
rz(-0.4131133123444961) q[7];
ry(2.522780299329674) q[8];
rz(2.4538999517428857) q[8];
ry(-2.0132210912976447) q[9];
rz(-1.5801412681564606) q[9];
ry(-2.681690384188067) q[10];
rz(-1.7146658227958778) q[10];
ry(2.646314304393281) q[11];
rz(0.7579455668583828) q[11];
ry(-2.9503631476338295) q[12];
rz(0.5359464915678008) q[12];
ry(-1.4360413478648741) q[13];
rz(-1.6200596639563396) q[13];
ry(1.2312946288854658) q[14];
rz(0.040277064589862153) q[14];
ry(-1.5984384270333187) q[15];
rz(-0.006072751461727143) q[15];
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
ry(3.105372662218143) q[0];
rz(0.5975807819650161) q[0];
ry(3.0767640723290888) q[1];
rz(-2.7847411792816614) q[1];
ry(-0.005867030618588119) q[2];
rz(0.8859020263279009) q[2];
ry(-1.6464770909375432) q[3];
rz(3.076518864810375) q[3];
ry(-1.5709419750041151) q[4];
rz(-1.032018541802337) q[4];
ry(2.0970871275619096) q[5];
rz(-2.083282928642208) q[5];
ry(3.141099934870657) q[6];
rz(-1.0891116818392073) q[6];
ry(-0.09332358070560426) q[7];
rz(-2.987461374061392) q[7];
ry(3.135635624328642) q[8];
rz(2.7674906677542244) q[8];
ry(0.00027842361350849387) q[9];
rz(1.9124839411837424) q[9];
ry(3.080301653061467) q[10];
rz(-1.8765148889545527) q[10];
ry(-0.004636600882392875) q[11];
rz(0.24003046038546408) q[11];
ry(-3.1355991461603145) q[12];
rz(-0.7380194590239091) q[12];
ry(1.587564286011582) q[13];
rz(0.9079431792439717) q[13];
ry(1.5605462211253047) q[14];
rz(-0.9903562771505746) q[14];
ry(-1.507754672645974) q[15];
rz(-3.1344597877722387) q[15];
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
ry(-2.813779059009342) q[0];
rz(0.8471794680489068) q[0];
ry(0.0008687258842785104) q[1];
rz(0.5684376964538863) q[1];
ry(-0.05772804362134709) q[2];
rz(-2.9567580105724267) q[2];
ry(1.5740286602096152) q[3];
rz(1.571136226004769) q[3];
ry(0.022028870957782942) q[4];
rz(-1.6352487945459275) q[4];
ry(1.5644079409554492) q[5];
rz(0.2525207144214343) q[5];
ry(1.4111711138333705) q[6];
rz(-0.6011371274167026) q[6];
ry(-1.4757336999799002) q[7];
rz(0.79136126838333) q[7];
ry(2.186554730967087) q[8];
rz(-0.5245762008722896) q[8];
ry(2.7766040104299616) q[9];
rz(1.2737638444316008) q[9];
ry(-2.1859532197757474) q[10];
rz(2.8929364465450162) q[10];
ry(-3.136104594961537) q[11];
rz(-0.030894273095510932) q[11];
ry(3.0848276437196445) q[12];
rz(-2.241355314084857) q[12];
ry(3.078299516670268) q[13];
rz(-2.241421302671944) q[13];
ry(2.905946813223515) q[14];
rz(2.1448888039142986) q[14];
ry(-1.553453656784524) q[15];
rz(2.644858525060695) q[15];
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
ry(1.6758261258206177) q[0];
rz(-0.032100351921236295) q[0];
ry(-0.004206883412632756) q[1];
rz(2.03114557072379) q[1];
ry(0.005956097466062182) q[2];
rz(0.8158424991123229) q[2];
ry(1.5709530308251782) q[3];
rz(-3.095728971829194) q[3];
ry(3.1400491023325348) q[4];
rz(1.420203628378714) q[4];
ry(0.00567460472990141) q[5];
rz(-0.06384896959711887) q[5];
ry(-0.0003719739565477117) q[6];
rz(-0.7731168144221021) q[6];
ry(-0.0023754043809240954) q[7];
rz(2.2673885630926196) q[7];
ry(3.107168384526362) q[8];
rz(-2.769406503713148) q[8];
ry(0.0002744551942293625) q[9];
rz(-0.4551866463246741) q[9];
ry(-1.5647690138457233) q[10];
rz(-1.5247675696593435) q[10];
ry(-0.0007228844688080827) q[11];
rz(-0.24258759478408945) q[11];
ry(3.1404732588201236) q[12];
rz(0.6399722315412917) q[12];
ry(-1.4368693149563878) q[13];
rz(1.1537316549772112) q[13];
ry(1.6670335801625809) q[14];
rz(0.8482847892374908) q[14];
ry(3.0252640097627426) q[15];
rz(-2.2934176907165384) q[15];
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
ry(0.23241216341770046) q[0];
rz(1.507396919182428) q[0];
ry(1.572485500684766) q[1];
rz(1.5347250598464097) q[1];
ry(-3.141289977413978) q[2];
rz(0.9987216972403282) q[2];
ry(-0.06519407189025196) q[3];
rz(1.936161825277198) q[3];
ry(-3.116716559403461) q[4];
rz(-2.333491103102459) q[4];
ry(2.986459511896035) q[5];
rz(2.7394020888654977) q[5];
ry(-2.265148098121905) q[6];
rz(-0.40710938263930596) q[6];
ry(-1.9346717154628568) q[7];
rz(-2.995960686022199) q[7];
ry(3.0898371484840026) q[8];
rz(1.721518923258728) q[8];
ry(-2.891911288116161) q[9];
rz(0.6465907155556377) q[9];
ry(0.33501174939946005) q[10];
rz(3.102217606686628) q[10];
ry(-3.0921934520849237) q[11];
rz(-2.388297333931422) q[11];
ry(2.0980313859389916) q[12];
rz(-1.0840533155466667) q[12];
ry(1.4996598690541219) q[13];
rz(-1.7717465138109034) q[13];
ry(-0.1624712527171091) q[14];
rz(2.8474453247261566) q[14];
ry(-1.3695088189849296) q[15];
rz(-1.5616278719533385) q[15];
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
ry(1.5750511413227966) q[0];
rz(1.5665373109362515) q[0];
ry(-0.12341178557507965) q[1];
rz(0.766220418268167) q[1];
ry(-1.5641362790527529) q[2];
rz(2.912161594214801) q[2];
ry(-0.021671620203952274) q[3];
rz(-1.9364975542860572) q[3];
ry(-3.13629539944016) q[4];
rz(-0.1017766710803196) q[4];
ry(-3.138821254431311) q[5];
rz(1.1380924678998523) q[5];
ry(-3.1405690731600715) q[6];
rz(-2.3135085698915594) q[6];
ry(-3.1412336158867036) q[7];
rz(-0.6866399039231116) q[7];
ry(-2.429743136750155) q[8];
rz(-2.844852638097105) q[8];
ry(3.1413227061738342) q[9];
rz(1.7373701070779763) q[9];
ry(0.5476888363678017) q[10];
rz(3.1110050977310415) q[10];
ry(-3.134018464055832) q[11];
rz(0.9847101277359168) q[11];
ry(3.1267532433828804) q[12];
rz(-1.1135497951052065) q[12];
ry(2.961999639873823) q[13];
rz(1.4328823245275721) q[13];
ry(3.126173255635515) q[14];
rz(-1.7642691097851815) q[14];
ry(0.30267867670713766) q[15];
rz(1.5951146472865698) q[15];
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
ry(1.5821329340933445) q[0];
rz(-1.5231939198010798) q[0];
ry(-1.326514585487042) q[1];
rz(-1.3068681372654831) q[1];
ry(1.4972709038246192) q[2];
rz(1.5755361691374612) q[2];
ry(-1.5295429886903698) q[3];
rz(1.4661830734340935) q[3];
ry(-0.7996076199850695) q[4];
rz(-2.696601224141376) q[4];
ry(-1.6825484756658957) q[5];
rz(-2.758180923361176) q[5];
ry(3.141021282950805) q[6];
rz(-2.795315670001042) q[6];
ry(2.3119335549835407) q[7];
rz(-2.9869248943504987) q[7];
ry(0.8586577524408208) q[8];
rz(2.718494638733011) q[8];
ry(-3.1410060940368063) q[9];
rz(0.789130717925715) q[9];
ry(2.8166507847795996) q[10];
rz(-0.00038756660484842465) q[10];
ry(-3.025696265786677) q[11];
rz(-0.5637173129152169) q[11];
ry(2.532362783361708) q[12];
rz(-0.40190778158784546) q[12];
ry(-0.5196159521979675) q[13];
rz(-1.0277672037669197) q[13];
ry(-1.44243356411726) q[14];
rz(1.6398773690482322) q[14];
ry(-1.7384278556780988) q[15];
rz(0.9751683402518987) q[15];
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
ry(-2.257862955237527) q[0];
rz(2.879304145813001) q[0];
ry(1.5425750565069958) q[1];
rz(3.0639579387803) q[1];
ry(-0.0010312161544135112) q[2];
rz(-0.028332099689814427) q[2];
ry(0.000768698384910671) q[3];
rz(0.9100591773499981) q[3];
ry(-0.00010915877353934178) q[4];
rz(0.15981621141007896) q[4];
ry(-3.141011238828139) q[5];
rz(-0.3942813817775923) q[5];
ry(3.140693151539978) q[6];
rz(-2.213940516159502) q[6];
ry(-3.1410778702578024) q[7];
rz(-1.6235477203244963) q[7];
ry(-2.4135995546678157) q[8];
rz(-2.773160618156491) q[8];
ry(0.0007777998909874739) q[9];
rz(0.07039248639233707) q[9];
ry(2.362568050604294) q[10];
rz(-1.469215794287046) q[10];
ry(-3.134962757893122) q[11];
rz(-0.47129432536879534) q[11];
ry(0.0023604460688213535) q[12];
rz(-1.5209305550083028) q[12];
ry(-0.05037978483785219) q[13];
rz(2.2483995964249113) q[13];
ry(-1.563946782542173) q[14];
rz(0.3251419959169759) q[14];
ry(-3.037793383666037) q[15];
rz(2.7032815229159675) q[15];
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
ry(-3.061533628775439) q[0];
rz(2.7882070775169354) q[0];
ry(2.072839540473936) q[1];
rz(-3.0516949423564754) q[1];
ry(2.748969870857267) q[2];
rz(-0.584608776035419) q[2];
ry(-1.0162482221940001) q[3];
rz(0.33513302629942654) q[3];
ry(-0.2766997762880002) q[4];
rz(-0.8978086818713304) q[4];
ry(2.474699596377196) q[5];
rz(-1.7087852015712772) q[5];
ry(-2.5027045288311522) q[6];
rz(0.42016638139388435) q[6];
ry(-2.525164298500359) q[7];
rz(0.24517762698262494) q[7];
ry(-0.9018845327294125) q[8];
rz(-1.9838533430197858) q[8];
ry(-0.9901317764463968) q[9];
rz(0.8591125169307849) q[9];
ry(-1.7004053231659397) q[10];
rz(0.6544967862033664) q[10];
ry(-3.0792509135511508) q[11];
rz(-0.5223909497702611) q[11];
ry(0.2373344943869915) q[12];
rz(-1.7664068452497892) q[12];
ry(0.5448941153783764) q[13];
rz(-2.3984167735586945) q[13];
ry(-0.8677917462982861) q[14];
rz(1.3840642086512913) q[14];
ry(-1.2336757523313147) q[15];
rz(-1.8122119208381031) q[15];
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
ry(0.9390306976400196) q[0];
rz(0.8104167227119695) q[0];
ry(-0.0665334925892564) q[1];
rz(1.507819111177641) q[1];
ry(-3.1406096326269575) q[2];
rz(-0.8913178735961056) q[2];
ry(-6.99856906853924e-05) q[3];
rz(0.26014061419477635) q[3];
ry(3.14066711906079) q[4];
rz(-1.6123439953483485) q[4];
ry(-3.1401225685997103) q[5];
rz(-1.740004134573677) q[5];
ry(0.0004937687988508799) q[6];
rz(0.4519252199227342) q[6];
ry(-3.137025174626844) q[7];
rz(-1.9047726088833254) q[7];
ry(0.006494076645876975) q[8];
rz(-2.5121301573253807) q[8];
ry(-0.003938832499217738) q[9];
rz(1.05262591012361) q[9];
ry(-1.5724583941280663) q[10];
rz(-1.8047905100285808) q[10];
ry(0.00213320299917306) q[11];
rz(3.098826388951262) q[11];
ry(-3.135883029781307) q[12];
rz(-0.030645084964011384) q[12];
ry(3.112935172933374) q[13];
rz(-2.328826370560671) q[13];
ry(-0.03661322465753696) q[14];
rz(-0.9372153483899366) q[14];
ry(-0.41450243066653564) q[15];
rz(2.2288199039392294) q[15];
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
ry(-1.3751098178835) q[0];
rz(2.878054773125607) q[0];
ry(-2.7607971780927807) q[1];
rz(0.3962977501003347) q[1];
ry(2.121622001679553) q[2];
rz(1.0394362209129726) q[2];
ry(0.23185311762037641) q[3];
rz(0.030486765426505788) q[3];
ry(-0.6552844742668507) q[4];
rz(-2.4914795536021073) q[4];
ry(-0.5389546176931885) q[5];
rz(0.0795940333329792) q[5];
ry(-0.34260229056001446) q[6];
rz(2.5947380207299315) q[6];
ry(-0.4987794344943346) q[7];
rz(2.248002492660741) q[7];
ry(1.3289662269226747) q[8];
rz(-0.31188421404246025) q[8];
ry(1.009195009848843) q[9];
rz(1.9415378272777426) q[9];
ry(1.6417471731073858) q[10];
rz(1.4071727549276938) q[10];
ry(1.0378354864256085) q[11];
rz(-3.1302706868792085) q[11];
ry(-1.829288510545088) q[12];
rz(-0.30218181792471693) q[12];
ry(-1.034528333467854) q[13];
rz(-0.2896124998820326) q[13];
ry(1.7205815485926461) q[14];
rz(3.07378740372259) q[14];
ry(-2.1462693167508764) q[15];
rz(0.3788970006228004) q[15];