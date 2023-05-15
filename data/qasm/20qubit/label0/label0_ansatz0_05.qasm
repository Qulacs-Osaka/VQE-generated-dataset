OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[0],q[1];
rz(-0.05771565532954802) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09770011395958558) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04229802752830778) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.06911935769738373) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.060790254595447846) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.07413844000855042) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.04175103617139575) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.0344198903791612) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.04472807367454436) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.09169437429260727) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.013633657163598783) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.09652359852470438) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.08422154419734658) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.07851769310245849) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.07973332693029818) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.07465746716050002) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.03837098122002723) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.02110967475153553) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.006962541910493912) q[19];
cx q[18],q[19];
h q[0];
rz(0.1369812193762581) q[0];
h q[0];
h q[1];
rz(1.592278606502305) q[1];
h q[1];
h q[2];
rz(-0.9392561137894279) q[2];
h q[2];
h q[3];
rz(1.2590524449164207) q[3];
h q[3];
h q[4];
rz(0.010568629965017296) q[4];
h q[4];
h q[5];
rz(1.3269376587718709) q[5];
h q[5];
h q[6];
rz(0.2974191203514841) q[6];
h q[6];
h q[7];
rz(1.192620135385507) q[7];
h q[7];
h q[8];
rz(0.9422241104346222) q[8];
h q[8];
h q[9];
rz(4.210125530516965e-05) q[9];
h q[9];
h q[10];
rz(0.8079678307045763) q[10];
h q[10];
h q[11];
rz(0.8227214308055872) q[11];
h q[11];
h q[12];
rz(1.5430571704363814) q[12];
h q[12];
h q[13];
rz(0.8930026101584346) q[13];
h q[13];
h q[14];
rz(0.5451573403273703) q[14];
h q[14];
h q[15];
rz(0.9959322300577647) q[15];
h q[15];
h q[16];
rz(0.5303566798201695) q[16];
h q[16];
h q[17];
rz(0.39706312726754817) q[17];
h q[17];
h q[18];
rz(1.5730530682306563) q[18];
h q[18];
h q[19];
rz(0.7759029210519155) q[19];
h q[19];
rz(0.17679385895071184) q[0];
rz(-0.6758917486442215) q[1];
rz(0.04131434547456391) q[2];
rz(-0.6619733421739585) q[3];
rz(0.05784549593645542) q[4];
rz(0.19390041188510762) q[5];
rz(0.19857981096760208) q[6];
rz(-0.19495728319783812) q[7];
rz(-0.7399484090104903) q[8];
rz(0.14507559746630197) q[9];
rz(0.5219014377600509) q[10];
rz(-0.4378267563606843) q[11];
rz(-0.8150881145537938) q[12];
rz(-0.733651277584521) q[13];
rz(0.038422228233489834) q[14];
rz(-0.06173095632266629) q[15];
rz(-0.14725416951018372) q[16];
rz(-0.320438076654669) q[17];
rz(-0.23003879693696994) q[18];
rz(-0.8350918149547616) q[19];
cx q[0],q[1];
rz(-0.2068926718822513) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.8742425131897367) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.37420533423294) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.23151161787231445) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.791397548068363) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.12399443219010857) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.1309692198232948) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.07356035920099299) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.10112691886889988) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.4068101281646508) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.005635369835261887) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.07933152738518585) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.5340770721690615) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.004159242483601255) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.24889948232880468) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.11960906455751916) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.2995355655595207) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.06336783389635875) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.9530569878531642) q[19];
cx q[18],q[19];
h q[0];
rz(-0.3109374284329208) q[0];
h q[0];
h q[1];
rz(0.6797324402023429) q[1];
h q[1];
h q[2];
rz(0.2658033371669974) q[2];
h q[2];
h q[3];
rz(0.46353124229907966) q[3];
h q[3];
h q[4];
rz(1.427062746132526) q[4];
h q[4];
h q[5];
rz(0.4509363804741017) q[5];
h q[5];
h q[6];
rz(1.1682311931814318) q[6];
h q[6];
h q[7];
rz(0.32213615328198386) q[7];
h q[7];
h q[8];
rz(-0.10361276294101521) q[8];
h q[8];
h q[9];
rz(0.04784804878225871) q[9];
h q[9];
h q[10];
rz(0.16675321325276196) q[10];
h q[10];
h q[11];
rz(0.7848603460858714) q[11];
h q[11];
h q[12];
rz(0.6030281935103707) q[12];
h q[12];
h q[13];
rz(0.7907230075972407) q[13];
h q[13];
h q[14];
rz(0.47571114528665237) q[14];
h q[14];
h q[15];
rz(0.5775097928436995) q[15];
h q[15];
h q[16];
rz(0.7640735010900556) q[16];
h q[16];
h q[17];
rz(-0.08431481065415446) q[17];
h q[17];
h q[18];
rz(0.01264199060398938) q[18];
h q[18];
h q[19];
rz(0.5762289449751452) q[19];
h q[19];
rz(0.4795280742128106) q[0];
rz(-0.9045753306195511) q[1];
rz(-0.012411696864960993) q[2];
rz(-0.6518918855852901) q[3];
rz(-1.2674119342389372) q[4];
rz(-0.8535349153334008) q[5];
rz(-1.1370614980852556) q[6];
rz(-0.970042429058843) q[7];
rz(-0.48351249479532055) q[8];
rz(0.5137795104553614) q[9];
rz(0.4434259721006969) q[10];
rz(-0.5729010606220607) q[11];
rz(-0.7057089137898428) q[12];
rz(-0.7788071820046139) q[13];
rz(-0.5226221348879249) q[14];
rz(-0.4284580715272678) q[15];
rz(-0.5015379012551946) q[16];
rz(0.18353875097086872) q[17];
rz(-0.3379644683217556) q[18];
rz(-0.846236883511827) q[19];
cx q[0],q[1];
rz(-0.05032432470730113) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.34500717029931255) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.31647468164749937) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.324338736618085) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.0790428279521107) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5891274743385154) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.15079815056753867) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.08044997278156946) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.27834119246531425) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.050873271425033825) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.9580227688423837) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.05098420231638648) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.03691058848323809) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.005483470273838787) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.8448155516131085) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.7918473598237031) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.13357372690152813) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.056635768855203356) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.5312827351717196) q[19];
cx q[18],q[19];
h q[0];
rz(0.1349824993521946) q[0];
h q[0];
h q[1];
rz(1.0244870054222976) q[1];
h q[1];
h q[2];
rz(-0.25040263425447973) q[2];
h q[2];
h q[3];
rz(0.29887154391458015) q[3];
h q[3];
h q[4];
rz(-0.6246827700937266) q[4];
h q[4];
h q[5];
rz(-0.9255782289086691) q[5];
h q[5];
h q[6];
rz(0.38485317432520166) q[6];
h q[6];
h q[7];
rz(0.012664533420198588) q[7];
h q[7];
h q[8];
rz(-0.6853533776200389) q[8];
h q[8];
h q[9];
rz(-0.7023966547285078) q[9];
h q[9];
h q[10];
rz(1.3757407950838292) q[10];
h q[10];
h q[11];
rz(0.3603707061541152) q[11];
h q[11];
h q[12];
rz(0.23555583276910963) q[12];
h q[12];
h q[13];
rz(0.4660562849745993) q[13];
h q[13];
h q[14];
rz(-0.38916068961594885) q[14];
h q[14];
h q[15];
rz(-0.014162063464896008) q[15];
h q[15];
h q[16];
rz(0.8291591874986142) q[16];
h q[16];
h q[17];
rz(0.22377615417380975) q[17];
h q[17];
h q[18];
rz(0.06582224736959756) q[18];
h q[18];
h q[19];
rz(0.7675822347728543) q[19];
h q[19];
rz(0.5196696726895138) q[0];
rz(0.07369518529954724) q[1];
rz(0.1955001700461261) q[2];
rz(-0.1789295335976896) q[3];
rz(-0.12644161297487627) q[4];
rz(-0.3197333637262627) q[5];
rz(-0.5280641835742687) q[6];
rz(-0.5733897474027627) q[7];
rz(0.27618415088026027) q[8];
rz(1.5140011699377491) q[9];
rz(-0.7230264874670294) q[10];
rz(-0.042049393605176455) q[11];
rz(0.37830248083349616) q[12];
rz(-0.1802996608964918) q[13];
rz(-0.04543487727627054) q[14];
rz(-0.1626674875818688) q[15];
rz(-0.41770898946113394) q[16];
rz(-0.39575445541349363) q[17];
rz(-0.1561305744862073) q[18];
rz(-0.6114399448054304) q[19];
cx q[0],q[1];
rz(-0.3214255849402848) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.5542925442278456) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.024653874512666034) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.41256714495876984) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.15475028035709362) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.3192539755279529) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.26692289562795196) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.0078066892804876204) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.06140018168729391) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(1.4843077152309812) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.9125322809484594) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.2595664187981474) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(0.9809680697696894) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.466352580902745) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-1.0769146159372027) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(1.0833912737769953) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.5714457079814472) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.003959397874648333) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(0.7117577456187969) q[19];
cx q[18],q[19];
h q[0];
rz(0.34558437362258937) q[0];
h q[0];
h q[1];
rz(0.6759022995986513) q[1];
h q[1];
h q[2];
rz(-0.9642303161424942) q[2];
h q[2];
h q[3];
rz(0.2438892891403284) q[3];
h q[3];
h q[4];
rz(-1.8905134193212558) q[4];
h q[4];
h q[5];
rz(-0.3165271082490937) q[5];
h q[5];
h q[6];
rz(-0.4674713503353726) q[6];
h q[6];
h q[7];
rz(-0.45469052487904543) q[7];
h q[7];
h q[8];
rz(0.7196778190282627) q[8];
h q[8];
h q[9];
rz(-1.5612418960761631) q[9];
h q[9];
h q[10];
rz(-0.09481200077975602) q[10];
h q[10];
h q[11];
rz(-0.5286484809846965) q[11];
h q[11];
h q[12];
rz(0.31564170055766566) q[12];
h q[12];
h q[13];
rz(0.0009645439286851268) q[13];
h q[13];
h q[14];
rz(-0.21806305716616536) q[14];
h q[14];
h q[15];
rz(-0.26407377407020216) q[15];
h q[15];
h q[16];
rz(1.2817227492920271) q[16];
h q[16];
h q[17];
rz(-0.8007690098351571) q[17];
h q[17];
h q[18];
rz(-0.05502511211184354) q[18];
h q[18];
h q[19];
rz(0.9195623385843708) q[19];
h q[19];
rz(0.6507207715539457) q[0];
rz(0.7448129939303852) q[1];
rz(1.2941659561592873) q[2];
rz(0.3001842053112745) q[3];
rz(-0.06304788717810708) q[4];
rz(-0.09612534239144) q[5];
rz(-0.16279981547326716) q[6];
rz(-0.13609437601486687) q[7];
rz(0.7619254319976434) q[8];
rz(-0.8167205085497535) q[9];
rz(-0.08456803099645858) q[10];
rz(0.029833203211349992) q[11];
rz(-0.014269529936540074) q[12];
rz(-0.16549779594839634) q[13];
rz(0.8976459555299516) q[14];
rz(-0.009830446078863483) q[15];
rz(0.4384478077960028) q[16];
rz(-0.3074559878316148) q[17];
rz(0.13595833410604877) q[18];
rz(-0.15470719130939045) q[19];
cx q[0],q[1];
rz(0.22483890025414252) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.6703511677737106) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.8490398283332627) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.11809258816861175) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.5439936086090593) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.6127233157162427) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.16728268439151214) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.008079518575267716) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.3296080341089034) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.013756046516362508) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.05551812109188025) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.4830082974775047) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.8311005507615968) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.5314551824043625) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.056474725182070015) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(0.4130043108602435) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.7280859622971444) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(0.049589624447108537) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.4565835665540924) q[19];
cx q[18],q[19];
h q[0];
rz(0.04594543618942172) q[0];
h q[0];
h q[1];
rz(0.19192028784476406) q[1];
h q[1];
h q[2];
rz(-0.610844746217326) q[2];
h q[2];
h q[3];
rz(-0.6910680197937477) q[3];
h q[3];
h q[4];
rz(0.1570682759699566) q[4];
h q[4];
h q[5];
rz(0.08136261151313406) q[5];
h q[5];
h q[6];
rz(-1.285072021959962) q[6];
h q[6];
h q[7];
rz(-0.26959027785211787) q[7];
h q[7];
h q[8];
rz(-1.9024999556851145) q[8];
h q[8];
h q[9];
rz(-1.6740604151267613) q[9];
h q[9];
h q[10];
rz(0.04031137121393545) q[10];
h q[10];
h q[11];
rz(-0.17561986387151132) q[11];
h q[11];
h q[12];
rz(0.39281520840413336) q[12];
h q[12];
h q[13];
rz(-0.16179583511042053) q[13];
h q[13];
h q[14];
rz(-1.0885137050978708) q[14];
h q[14];
h q[15];
rz(0.6716726552291328) q[15];
h q[15];
h q[16];
rz(-0.1161961482221979) q[16];
h q[16];
h q[17];
rz(-0.8749165923033756) q[17];
h q[17];
h q[18];
rz(-0.0456234081318457) q[18];
h q[18];
h q[19];
rz(0.5480851007242674) q[19];
h q[19];
rz(1.048311148099055) q[0];
rz(0.5540632936253341) q[1];
rz(-0.045833929603752994) q[2];
rz(-0.03222356117049411) q[3];
rz(0.292368492319955) q[4];
rz(0.15173498833457041) q[5];
rz(0.01621242144361602) q[6];
rz(0.19039324365097574) q[7];
rz(1.1569702856595157) q[8];
rz(-0.14237912742844908) q[9];
rz(0.2201398559547088) q[10];
rz(-0.1401981101671989) q[11];
rz(-0.09680106364830462) q[12];
rz(0.06826807924469258) q[13];
rz(0.34216782777962806) q[14];
rz(-0.014345853534879318) q[15];
rz(-0.38387546991009236) q[16];
rz(1.1807116934283406) q[17];
rz(0.003717643835732963) q[18];
rz(-0.11198456449522615) q[19];
cx q[0],q[1];
rz(0.5039909142093751) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.7733903679409576) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.7457980365588075) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.6207072854000393) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.42280271212654297) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.8216214280754686) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5327666998037565) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.16783641132366475) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.007849985607842759) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.04346252520642343) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.6916322949739596) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.25830448293979275) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.2414460308947212) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(0.02632874231965352) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.5175616757501473) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.4386605172750189) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.15216542588841417) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.14571257678529964) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.011540288352843557) q[19];
cx q[18],q[19];
h q[0];
rz(-0.5144404599686005) q[0];
h q[0];
h q[1];
rz(-0.041244162075319335) q[1];
h q[1];
h q[2];
rz(-0.6551332107519888) q[2];
h q[2];
h q[3];
rz(-1.6332827962980887) q[3];
h q[3];
h q[4];
rz(-0.4270582584470733) q[4];
h q[4];
h q[5];
rz(1.0368945324022703) q[5];
h q[5];
h q[6];
rz(-0.8066895197338401) q[6];
h q[6];
h q[7];
rz(-0.29421078385903543) q[7];
h q[7];
h q[8];
rz(-0.26111776964810507) q[8];
h q[8];
h q[9];
rz(-1.8019291293912139) q[9];
h q[9];
h q[10];
rz(-1.039595191919606) q[10];
h q[10];
h q[11];
rz(-0.3712998967230169) q[11];
h q[11];
h q[12];
rz(0.05427189264672279) q[12];
h q[12];
h q[13];
rz(-1.6929189752829505) q[13];
h q[13];
h q[14];
rz(0.1512458549279813) q[14];
h q[14];
h q[15];
rz(-1.1291027304913424) q[15];
h q[15];
h q[16];
rz(-0.4537103969531532) q[16];
h q[16];
h q[17];
rz(-0.7083816619316974) q[17];
h q[17];
h q[18];
rz(-1.7717181547721903) q[18];
h q[18];
h q[19];
rz(0.3858142453123872) q[19];
h q[19];
rz(1.2533203590215478) q[0];
rz(-0.34037779966267556) q[1];
rz(0.24550699569149223) q[2];
rz(-0.008144657057831944) q[3];
rz(0.008417911128436064) q[4];
rz(-0.02408293535488059) q[5];
rz(-0.013390642469267765) q[6];
rz(0.001943259185907541) q[7];
rz(0.48415356982891006) q[8];
rz(-0.32450269484285793) q[9];
rz(-0.0031883581105481714) q[10];
rz(0.06879009109588714) q[11];
rz(0.10900128577385905) q[12];
rz(-0.04729527820647019) q[13];
rz(0.8318497349003028) q[14];
rz(-0.0022108136402423416) q[15];
rz(0.0032622052010059387) q[16];
rz(0.2843975225726804) q[17];
rz(-0.005104760314915978) q[18];
rz(-0.06251161540579633) q[19];
cx q[0],q[1];
rz(-1.251403396688168) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-1.0005768106081352) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.28028683814862104) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.400385475039573) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.19517189553934655) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.014808136653691645) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.8207731368689464) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.10139910480267958) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.9589188080263769) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.20425122848052957) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.039942747912165635) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.07716869772989302) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(1.0162820223784117) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.33423592768777977) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.21290760767878397) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.9711838177409338) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.15158906908437092) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(1.014172115747066) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.7946081080995671) q[19];
cx q[18],q[19];
h q[0];
rz(-1.2443711323564943) q[0];
h q[0];
h q[1];
rz(0.24284646247037922) q[1];
h q[1];
h q[2];
rz(0.18195236718471458) q[2];
h q[2];
h q[3];
rz(-2.2970232256264675) q[3];
h q[3];
h q[4];
rz(0.2543619926738577) q[4];
h q[4];
h q[5];
rz(-2.6455786130048367) q[5];
h q[5];
h q[6];
rz(-1.6236871709628546) q[6];
h q[6];
h q[7];
rz(-1.4238509646675936) q[7];
h q[7];
h q[8];
rz(0.04634842778236019) q[8];
h q[8];
h q[9];
rz(-1.4467052880608713) q[9];
h q[9];
h q[10];
rz(-2.0082883485187213) q[10];
h q[10];
h q[11];
rz(-1.7954890185365548) q[11];
h q[11];
h q[12];
rz(0.5511852496519732) q[12];
h q[12];
h q[13];
rz(-0.16504864485912532) q[13];
h q[13];
h q[14];
rz(-1.0355312630408515) q[14];
h q[14];
h q[15];
rz(-1.6186511752569477) q[15];
h q[15];
h q[16];
rz(1.4119686646318614) q[16];
h q[16];
h q[17];
rz(-0.005516775262643947) q[17];
h q[17];
h q[18];
rz(-0.18972382183242217) q[18];
h q[18];
h q[19];
rz(-0.2554924262190797) q[19];
h q[19];
rz(0.3560893270994361) q[0];
rz(0.1119170549997493) q[1];
rz(-0.025856601710001488) q[2];
rz(-0.06406133939356935) q[3];
rz(0.022527860079956517) q[4];
rz(-0.023264116388654298) q[5];
rz(0.006089443279758612) q[6];
rz(-0.3872871757653399) q[7];
rz(-1.1579381070450663) q[8];
rz(-0.019188238144944397) q[9];
rz(1.1054856726352487) q[10];
rz(-0.03774027475220052) q[11];
rz(0.3028458326611467) q[12];
rz(0.03436001458988438) q[13];
rz(0.10020316948832904) q[14];
rz(0.009934650657942174) q[15];
rz(-0.027902759899861396) q[16];
rz(0.16847313491864696) q[17];
rz(-0.0032132989814525796) q[18];
rz(-0.03393348797934819) q[19];
cx q[0],q[1];
rz(0.34562630197378275) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.15330303889229874) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.9325868278504478) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.7318895069719122) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.14801609457745413) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(1.5384687801491728) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.3500803642668761) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.0029138582805951123) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.2839440512368748) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.23871883459324844) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.09950508948385216) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.27469473969689834) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.9304772578551578) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.022490571231114358) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.36990515245180283) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.9124134411160713) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.2524224497610017) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-1.3297641135916438) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(1.0276098054423326) q[19];
cx q[18],q[19];
h q[0];
rz(0.037352961414578556) q[0];
h q[0];
h q[1];
rz(-0.2643329798111611) q[1];
h q[1];
h q[2];
rz(-0.2889845856607203) q[2];
h q[2];
h q[3];
rz(0.38013699893348246) q[3];
h q[3];
h q[4];
rz(0.005496216186948802) q[4];
h q[4];
h q[5];
rz(-1.5049892687107966) q[5];
h q[5];
h q[6];
rz(-1.5671395924410223) q[6];
h q[6];
h q[7];
rz(-0.0067243208751639185) q[7];
h q[7];
h q[8];
rz(0.034830597604789536) q[8];
h q[8];
h q[9];
rz(-1.6026202298045868) q[9];
h q[9];
h q[10];
rz(-0.004783594817257196) q[10];
h q[10];
h q[11];
rz(-1.0254157103137858) q[11];
h q[11];
h q[12];
rz(0.00784206118905413) q[12];
h q[12];
h q[13];
rz(0.035368185099224854) q[13];
h q[13];
h q[14];
rz(-0.40883312873334404) q[14];
h q[14];
h q[15];
rz(1.945399683820112) q[15];
h q[15];
h q[16];
rz(-0.832602465929265) q[16];
h q[16];
h q[17];
rz(0.02942084232134007) q[17];
h q[17];
h q[18];
rz(-1.6821094349529855) q[18];
h q[18];
h q[19];
rz(-1.4901225506834455) q[19];
h q[19];
rz(1.1990449209567178) q[0];
rz(-0.10273442019303981) q[1];
rz(-0.3259946721733949) q[2];
rz(0.017332346667705275) q[3];
rz(0.022751807510123713) q[4];
rz(0.018515015897963134) q[5];
rz(0.03513622115780843) q[6];
rz(0.374808581130239) q[7];
rz(0.8176969032049578) q[8];
rz(3.084050171103271) q[9];
rz(1.9985923572081157) q[10];
rz(-0.08658623966501107) q[11];
rz(-0.24888580078930778) q[12];
rz(-0.02701234111643121) q[13];
rz(-0.17928824084635375) q[14];
rz(0.0711624485193699) q[15];
rz(-0.005749052106279706) q[16];
rz(-0.04075240091756079) q[17];
rz(-0.01313404104261128) q[18];
rz(-0.011312280264188642) q[19];