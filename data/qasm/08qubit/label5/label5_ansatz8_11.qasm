OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.4535107206287003) q[0];
ry(2.548986910129094) q[1];
cx q[0],q[1];
ry(2.634606273253029) q[0];
ry(-0.932318900913515) q[1];
cx q[0],q[1];
ry(-2.2546447405672323) q[2];
ry(0.4501414225371948) q[3];
cx q[2],q[3];
ry(3.0219039031228414) q[2];
ry(-0.5702414822737586) q[3];
cx q[2],q[3];
ry(1.4779159100924204) q[4];
ry(-1.2167277266697036) q[5];
cx q[4],q[5];
ry(-1.224968922529764) q[4];
ry(2.8950959401331486) q[5];
cx q[4],q[5];
ry(-0.05687241132403359) q[6];
ry(1.0158445442288873) q[7];
cx q[6],q[7];
ry(2.4508770722402913) q[6];
ry(2.4537793756210093) q[7];
cx q[6],q[7];
ry(-0.16171015037240774) q[0];
ry(2.6911067466028644) q[2];
cx q[0],q[2];
ry(-2.729861616492021) q[0];
ry(-2.2658907458543185) q[2];
cx q[0],q[2];
ry(-0.7253215279275872) q[2];
ry(-1.3176795097634535) q[4];
cx q[2],q[4];
ry(-0.4255529354107743) q[2];
ry(-1.366658324940474) q[4];
cx q[2],q[4];
ry(-0.11673651159673515) q[4];
ry(-2.932484315374523) q[6];
cx q[4],q[6];
ry(2.4919059299546125) q[4];
ry(1.4561809596167983) q[6];
cx q[4],q[6];
ry(1.2788463293262557) q[1];
ry(-2.617612809842423) q[3];
cx q[1],q[3];
ry(1.5077996629097667) q[1];
ry(1.9985803498899257) q[3];
cx q[1],q[3];
ry(-2.951060734607674) q[3];
ry(-1.7499417017809897) q[5];
cx q[3],q[5];
ry(1.9037947474681935) q[3];
ry(2.5211841783263904) q[5];
cx q[3],q[5];
ry(2.4941226295595564) q[5];
ry(-1.9857924449686903) q[7];
cx q[5],q[7];
ry(-0.01679888541377516) q[5];
ry(-1.8456340027219742) q[7];
cx q[5],q[7];
ry(0.14190812064494907) q[0];
ry(0.7941675352388866) q[1];
cx q[0],q[1];
ry(-0.7865566431750137) q[0];
ry(2.736575361787619) q[1];
cx q[0],q[1];
ry(0.22540609636103276) q[2];
ry(-0.5881332534331655) q[3];
cx q[2],q[3];
ry(0.7971641467079267) q[2];
ry(-0.8405917360397694) q[3];
cx q[2],q[3];
ry(-1.0217597746938492) q[4];
ry(-0.924431616267471) q[5];
cx q[4],q[5];
ry(1.5809798712473473) q[4];
ry(2.04732330418173) q[5];
cx q[4],q[5];
ry(-0.6499071975178872) q[6];
ry(0.2259749484871039) q[7];
cx q[6],q[7];
ry(1.9985024587934372) q[6];
ry(-1.1324348622600542) q[7];
cx q[6],q[7];
ry(2.1367567365606717) q[0];
ry(0.3534984142451121) q[2];
cx q[0],q[2];
ry(2.2410206114185653) q[0];
ry(-2.4689522131319643) q[2];
cx q[0],q[2];
ry(2.5738918172827625) q[2];
ry(-2.354908004987067) q[4];
cx q[2],q[4];
ry(2.1260951172977447) q[2];
ry(1.2511276661677044) q[4];
cx q[2],q[4];
ry(-1.1478254939931594) q[4];
ry(0.4695447330214428) q[6];
cx q[4],q[6];
ry(1.0989703817123015) q[4];
ry(-0.8500605029501732) q[6];
cx q[4],q[6];
ry(-0.0888879676044286) q[1];
ry(1.2728153579271577) q[3];
cx q[1],q[3];
ry(1.6174281697152793) q[1];
ry(-0.18114918451673567) q[3];
cx q[1],q[3];
ry(1.1701370162694424) q[3];
ry(-1.6258442110557765) q[5];
cx q[3],q[5];
ry(-0.8149374933004658) q[3];
ry(-2.616894506452441) q[5];
cx q[3],q[5];
ry(2.312511661256043) q[5];
ry(0.42649459791077104) q[7];
cx q[5],q[7];
ry(2.6671231160322795) q[5];
ry(1.088359794304391) q[7];
cx q[5],q[7];
ry(2.487581666757829) q[0];
ry(1.4901652619553953) q[1];
cx q[0],q[1];
ry(-2.8143290758461235) q[0];
ry(0.5083393981582449) q[1];
cx q[0],q[1];
ry(3.031654747253754) q[2];
ry(-1.3422109467275947) q[3];
cx q[2],q[3];
ry(1.172488589137644) q[2];
ry(-2.1287843328260623) q[3];
cx q[2],q[3];
ry(-1.8940500954674393) q[4];
ry(0.3947166582407542) q[5];
cx q[4],q[5];
ry(-2.2354906376029366) q[4];
ry(2.1196781165333487) q[5];
cx q[4],q[5];
ry(-2.3679926489625425) q[6];
ry(1.0892618046804428) q[7];
cx q[6],q[7];
ry(1.9766340916468723) q[6];
ry(0.26871080803874525) q[7];
cx q[6],q[7];
ry(1.397520651739683) q[0];
ry(-1.4454570086022835) q[2];
cx q[0],q[2];
ry(-1.239597033733098) q[0];
ry(0.7686819674193659) q[2];
cx q[0],q[2];
ry(0.9783240266022403) q[2];
ry(0.5246479272027091) q[4];
cx q[2],q[4];
ry(-1.9069593337415531) q[2];
ry(3.0643103608812403) q[4];
cx q[2],q[4];
ry(-0.26469331541008323) q[4];
ry(-0.22297283217672775) q[6];
cx q[4],q[6];
ry(-1.0500416735927451) q[4];
ry(-0.7594632608531855) q[6];
cx q[4],q[6];
ry(1.6608875459521684) q[1];
ry(2.8517311242012364) q[3];
cx q[1],q[3];
ry(1.6465208241235034) q[1];
ry(-3.066599486008555) q[3];
cx q[1],q[3];
ry(-1.6341286177731034) q[3];
ry(2.893066834284435) q[5];
cx q[3],q[5];
ry(3.016550308212688) q[3];
ry(-0.15870202219381688) q[5];
cx q[3],q[5];
ry(3.0331266771226537) q[5];
ry(-0.05761466444853447) q[7];
cx q[5],q[7];
ry(-0.3649538707673895) q[5];
ry(-1.0723814374987493) q[7];
cx q[5],q[7];
ry(2.5910300843011735) q[0];
ry(-1.135970047153839) q[1];
cx q[0],q[1];
ry(-2.140747998026386) q[0];
ry(-0.6540952764702057) q[1];
cx q[0],q[1];
ry(1.9878796959282425) q[2];
ry(-0.33543053837786463) q[3];
cx q[2],q[3];
ry(-2.756792244696214) q[2];
ry(-0.3709468540898145) q[3];
cx q[2],q[3];
ry(1.2723778291968426) q[4];
ry(-0.286487440429128) q[5];
cx q[4],q[5];
ry(0.39311756841319406) q[4];
ry(1.1244446543581845) q[5];
cx q[4],q[5];
ry(2.830335522255373) q[6];
ry(-0.9494364365987794) q[7];
cx q[6],q[7];
ry(-2.6285120184099378) q[6];
ry(-1.9311503091048614) q[7];
cx q[6],q[7];
ry(1.672638389382229) q[0];
ry(3.0950978805925717) q[2];
cx q[0],q[2];
ry(1.364622561415809) q[0];
ry(2.0775324008369864) q[2];
cx q[0],q[2];
ry(-2.0148119120089034) q[2];
ry(1.6444019449861234) q[4];
cx q[2],q[4];
ry(1.9699198365307964) q[2];
ry(1.027139986599983) q[4];
cx q[2],q[4];
ry(-0.7207974602889375) q[4];
ry(-1.3416862352268222) q[6];
cx q[4],q[6];
ry(0.40626805002791233) q[4];
ry(2.9577854366717053) q[6];
cx q[4],q[6];
ry(-1.8564843666726276) q[1];
ry(-0.08094825329124422) q[3];
cx q[1],q[3];
ry(3.0353224477508682) q[1];
ry(-2.8576049705413054) q[3];
cx q[1],q[3];
ry(-1.6218177189462262) q[3];
ry(-1.415645848823348) q[5];
cx q[3],q[5];
ry(2.9965986762365664) q[3];
ry(-0.011228935933321006) q[5];
cx q[3],q[5];
ry(0.3710371716336921) q[5];
ry(1.9212780701511296) q[7];
cx q[5],q[7];
ry(-2.565684703303424) q[5];
ry(0.8536092889903237) q[7];
cx q[5],q[7];
ry(-0.31500380359223346) q[0];
ry(-0.3352659308133477) q[1];
cx q[0],q[1];
ry(0.887928799728461) q[0];
ry(-0.6978658365855885) q[1];
cx q[0],q[1];
ry(0.7798252001451589) q[2];
ry(0.06438941350505978) q[3];
cx q[2],q[3];
ry(-1.8066603642048324) q[2];
ry(0.7309511767024768) q[3];
cx q[2],q[3];
ry(1.624113483823071) q[4];
ry(0.8217756108949212) q[5];
cx q[4],q[5];
ry(2.305390135539408) q[4];
ry(-0.5650746111837672) q[5];
cx q[4],q[5];
ry(2.5331351264009654) q[6];
ry(-1.0903341000628748) q[7];
cx q[6],q[7];
ry(-1.4437496027533312) q[6];
ry(-0.8159302806220233) q[7];
cx q[6],q[7];
ry(2.04231485415532) q[0];
ry(1.5746877077329842) q[2];
cx q[0],q[2];
ry(-0.6627484225293275) q[0];
ry(2.6721287313135207) q[2];
cx q[0],q[2];
ry(-2.1790142991068797) q[2];
ry(-2.9036127672441365) q[4];
cx q[2],q[4];
ry(3.1415699295511437) q[2];
ry(-3.0990691052233403) q[4];
cx q[2],q[4];
ry(-0.6278385679743923) q[4];
ry(-1.1115783152551424) q[6];
cx q[4],q[6];
ry(1.2241845202957327) q[4];
ry(0.01712215424984187) q[6];
cx q[4],q[6];
ry(0.5232842918146758) q[1];
ry(-2.432992538460148) q[3];
cx q[1],q[3];
ry(-1.4818298267680656) q[1];
ry(0.8850713096018019) q[3];
cx q[1],q[3];
ry(-2.513281564205383) q[3];
ry(-1.64865687918838) q[5];
cx q[3],q[5];
ry(-2.636282588997077) q[3];
ry(1.479794726963088) q[5];
cx q[3],q[5];
ry(-0.9576713671793797) q[5];
ry(2.041544598183685) q[7];
cx q[5],q[7];
ry(-1.0621020450360978) q[5];
ry(-0.6399506190429841) q[7];
cx q[5],q[7];
ry(0.814397670406149) q[0];
ry(2.030152732224983) q[1];
cx q[0],q[1];
ry(-2.8182658962817695) q[0];
ry(-2.086160192047406) q[1];
cx q[0],q[1];
ry(-2.027664562734527) q[2];
ry(-2.2123957599518382) q[3];
cx q[2],q[3];
ry(-0.663671188216775) q[2];
ry(2.0599185794197563) q[3];
cx q[2],q[3];
ry(2.0690192482746967) q[4];
ry(-2.758327122951029) q[5];
cx q[4],q[5];
ry(-0.9325495225817585) q[4];
ry(-0.7395851025958636) q[5];
cx q[4],q[5];
ry(-0.042500306738253) q[6];
ry(1.8080532923396107) q[7];
cx q[6],q[7];
ry(-0.9722626599646542) q[6];
ry(2.7969163488998943) q[7];
cx q[6],q[7];
ry(-1.0109555556893026) q[0];
ry(1.8900282039189855) q[2];
cx q[0],q[2];
ry(-3.0245511835825356) q[0];
ry(0.8543181284033938) q[2];
cx q[0],q[2];
ry(0.19121505139783856) q[2];
ry(0.49623856343574513) q[4];
cx q[2],q[4];
ry(-1.8554659400624582) q[2];
ry(-1.585854494073037) q[4];
cx q[2],q[4];
ry(0.6677932567993167) q[4];
ry(3.0482823097476657) q[6];
cx q[4],q[6];
ry(-1.8690717041906555) q[4];
ry(-1.1039994213350774) q[6];
cx q[4],q[6];
ry(2.3762861461962443) q[1];
ry(2.4260524806892474) q[3];
cx q[1],q[3];
ry(1.154178926915269) q[1];
ry(-1.520972053350274) q[3];
cx q[1],q[3];
ry(-0.45502250841471437) q[3];
ry(-0.7364478651616448) q[5];
cx q[3],q[5];
ry(1.2125969318564493) q[3];
ry(2.3094699261731058) q[5];
cx q[3],q[5];
ry(-0.3451423063061459) q[5];
ry(1.8699429849754976) q[7];
cx q[5],q[7];
ry(0.2968777113872374) q[5];
ry(0.5785354851422895) q[7];
cx q[5],q[7];
ry(1.6200543391551707) q[0];
ry(1.2680778315341663) q[1];
cx q[0],q[1];
ry(-1.6257909639053871) q[0];
ry(-2.108825946517114) q[1];
cx q[0],q[1];
ry(-0.42776472448839375) q[2];
ry(1.2973475791539395) q[3];
cx q[2],q[3];
ry(-0.4424475851911615) q[2];
ry(-0.4540548984056869) q[3];
cx q[2],q[3];
ry(-2.725657696211528) q[4];
ry(-0.2448349479420761) q[5];
cx q[4],q[5];
ry(2.905455422657464) q[4];
ry(-0.23761927154956997) q[5];
cx q[4],q[5];
ry(-2.9231861129585917) q[6];
ry(-0.4086686002941546) q[7];
cx q[6],q[7];
ry(1.0879682246762463) q[6];
ry(-2.421480034909893) q[7];
cx q[6],q[7];
ry(1.9971639167591846) q[0];
ry(-0.6608219628124159) q[2];
cx q[0],q[2];
ry(-2.856423950227809) q[0];
ry(-2.9769774035568703) q[2];
cx q[0],q[2];
ry(0.13217266748652706) q[2];
ry(2.8055065885425994) q[4];
cx q[2],q[4];
ry(-2.246403124223349) q[2];
ry(-1.2760319767688235) q[4];
cx q[2],q[4];
ry(-1.8497150382044538) q[4];
ry(0.5550714007141898) q[6];
cx q[4],q[6];
ry(1.0310104780818097) q[4];
ry(-0.35660245894421116) q[6];
cx q[4],q[6];
ry(-2.521733544801403) q[1];
ry(2.147559555732844) q[3];
cx q[1],q[3];
ry(1.545478698757726) q[1];
ry(2.975951725474121) q[3];
cx q[1],q[3];
ry(3.043521705871936) q[3];
ry(-2.063052635001797) q[5];
cx q[3],q[5];
ry(-2.997541658910042) q[3];
ry(-0.1072017335369851) q[5];
cx q[3],q[5];
ry(-1.4304458970479343) q[5];
ry(-0.33185142295539327) q[7];
cx q[5],q[7];
ry(-1.0292602145332106) q[5];
ry(2.4649159633405753) q[7];
cx q[5],q[7];
ry(2.5219616057002003) q[0];
ry(1.1426045233139155) q[1];
cx q[0],q[1];
ry(2.9823845649671394) q[0];
ry(-2.4324209108337342) q[1];
cx q[0],q[1];
ry(-2.520477106967457) q[2];
ry(1.4710316020904592) q[3];
cx q[2],q[3];
ry(-0.701799546233338) q[2];
ry(-0.638377223629264) q[3];
cx q[2],q[3];
ry(-2.3364516094372663) q[4];
ry(-1.0419722868176986) q[5];
cx q[4],q[5];
ry(-2.1580473287897437) q[4];
ry(1.6850091976881068) q[5];
cx q[4],q[5];
ry(0.20769667679861628) q[6];
ry(2.50521423685862) q[7];
cx q[6],q[7];
ry(-1.6437122243480404) q[6];
ry(-0.08013529268342623) q[7];
cx q[6],q[7];
ry(-2.765486163951716) q[0];
ry(1.3028833860942024) q[2];
cx q[0],q[2];
ry(2.7066953395006506) q[0];
ry(1.8793304327484732) q[2];
cx q[0],q[2];
ry(-3.0954902234087016) q[2];
ry(2.086477349165962) q[4];
cx q[2],q[4];
ry(-2.2104065691544497) q[2];
ry(2.839772523009981) q[4];
cx q[2],q[4];
ry(0.39014740028703715) q[4];
ry(-0.10734298161136203) q[6];
cx q[4],q[6];
ry(-1.642961695994544) q[4];
ry(-2.914512307361188) q[6];
cx q[4],q[6];
ry(-0.17626134434745122) q[1];
ry(2.1442843090080776) q[3];
cx q[1],q[3];
ry(2.9786792446249817) q[1];
ry(2.42729245380524) q[3];
cx q[1],q[3];
ry(-0.2238272863351568) q[3];
ry(0.20209569393984417) q[5];
cx q[3],q[5];
ry(0.6170741368122714) q[3];
ry(-0.1187106432917515) q[5];
cx q[3],q[5];
ry(1.9887509410258293) q[5];
ry(-0.5689664410089402) q[7];
cx q[5],q[7];
ry(-1.2461860633665234) q[5];
ry(1.947207017945814) q[7];
cx q[5],q[7];
ry(2.0101179160509286) q[0];
ry(2.4465491956481187) q[1];
cx q[0],q[1];
ry(-1.5545763117813156) q[0];
ry(1.6893823923953029) q[1];
cx q[0],q[1];
ry(-2.218422377186661) q[2];
ry(-1.2010208452760192) q[3];
cx q[2],q[3];
ry(-1.5616658772509704) q[2];
ry(-1.7679871591424567) q[3];
cx q[2],q[3];
ry(-1.4262521214663435) q[4];
ry(2.3462279830687245) q[5];
cx q[4],q[5];
ry(-0.06641118858245498) q[4];
ry(1.8981649562692382) q[5];
cx q[4],q[5];
ry(-2.2313132489992658) q[6];
ry(0.6930737996243206) q[7];
cx q[6],q[7];
ry(0.49726756169664793) q[6];
ry(1.2584011016233712) q[7];
cx q[6],q[7];
ry(2.4462713640945304) q[0];
ry(0.1747379475202623) q[2];
cx q[0],q[2];
ry(0.9702212924791581) q[0];
ry(1.6767442405477173) q[2];
cx q[0],q[2];
ry(-2.142920478435724) q[2];
ry(-0.2694578442699872) q[4];
cx q[2],q[4];
ry(-2.2376226875998455) q[2];
ry(2.5115867936376826) q[4];
cx q[2],q[4];
ry(1.3770354745037763) q[4];
ry(-1.9413411990839422) q[6];
cx q[4],q[6];
ry(0.7367793638958293) q[4];
ry(-1.8927320308200104) q[6];
cx q[4],q[6];
ry(-2.3244287310382337) q[1];
ry(-1.2528147996221328) q[3];
cx q[1],q[3];
ry(-0.9835057565845631) q[1];
ry(1.4278534492218489) q[3];
cx q[1],q[3];
ry(-1.4401348278854877) q[3];
ry(2.6041600443018944) q[5];
cx q[3],q[5];
ry(-2.100518664744289) q[3];
ry(-0.39975557760071023) q[5];
cx q[3],q[5];
ry(-0.9946120200802318) q[5];
ry(1.3146568827243863) q[7];
cx q[5],q[7];
ry(-2.09200720433518) q[5];
ry(-2.0972551775872814) q[7];
cx q[5],q[7];
ry(0.5400992711218212) q[0];
ry(-1.6092907649436896) q[1];
cx q[0],q[1];
ry(2.578237602712347) q[0];
ry(1.273169153506072) q[1];
cx q[0],q[1];
ry(2.7209177885544875) q[2];
ry(0.473614984238777) q[3];
cx q[2],q[3];
ry(2.9536775954876218) q[2];
ry(1.7442506005018805) q[3];
cx q[2],q[3];
ry(0.24181928309618111) q[4];
ry(1.6260711243486297) q[5];
cx q[4],q[5];
ry(3.102706419898207) q[4];
ry(-1.3362807179529872) q[5];
cx q[4],q[5];
ry(-1.6609657063528376) q[6];
ry(2.657647254357565) q[7];
cx q[6],q[7];
ry(1.9220426926627536) q[6];
ry(1.188743314987546) q[7];
cx q[6],q[7];
ry(0.6408028278235255) q[0];
ry(-1.8381261172569268) q[2];
cx q[0],q[2];
ry(-0.5038651219483841) q[0];
ry(-1.7562411833208582) q[2];
cx q[0],q[2];
ry(1.6076472433070779) q[2];
ry(1.704952849393765) q[4];
cx q[2],q[4];
ry(-0.2322268366950279) q[2];
ry(2.8619171315232284) q[4];
cx q[2],q[4];
ry(2.1226661185815217) q[4];
ry(2.259345111950254) q[6];
cx q[4],q[6];
ry(-1.8057154964876367) q[4];
ry(-2.015656083499842) q[6];
cx q[4],q[6];
ry(-0.7897329014005555) q[1];
ry(-0.5225512817861375) q[3];
cx q[1],q[3];
ry(-2.232895259841519) q[1];
ry(-0.4357965881837091) q[3];
cx q[1],q[3];
ry(0.026558475635141927) q[3];
ry(-2.036372723834315) q[5];
cx q[3],q[5];
ry(-0.5274619496066739) q[3];
ry(-1.863339126067217) q[5];
cx q[3],q[5];
ry(0.7855168394219852) q[5];
ry(-0.6621116697586263) q[7];
cx q[5],q[7];
ry(-1.936201841947391) q[5];
ry(-2.3149009766876634) q[7];
cx q[5],q[7];
ry(0.5886702160219789) q[0];
ry(2.6548701378935555) q[1];
cx q[0],q[1];
ry(-2.1130701371762983) q[0];
ry(-0.4568212353477416) q[1];
cx q[0],q[1];
ry(1.2091409484365618) q[2];
ry(0.9104663294011459) q[3];
cx q[2],q[3];
ry(-1.898589772081153) q[2];
ry(-2.7067762548704115) q[3];
cx q[2],q[3];
ry(-2.433348328097703) q[4];
ry(-2.723972165307831) q[5];
cx q[4],q[5];
ry(-1.3659938247654289) q[4];
ry(-0.13721652325770428) q[5];
cx q[4],q[5];
ry(-1.6324264112903724) q[6];
ry(2.1607579816763636) q[7];
cx q[6],q[7];
ry(2.3398122565446204) q[6];
ry(3.0750768260674834) q[7];
cx q[6],q[7];
ry(-0.8913844008304715) q[0];
ry(1.2240464626034298) q[2];
cx q[0],q[2];
ry(-0.9104003954555208) q[0];
ry(2.8540671753088516) q[2];
cx q[0],q[2];
ry(-0.14025981174368596) q[2];
ry(1.080731765030542) q[4];
cx q[2],q[4];
ry(-0.222697556912617) q[2];
ry(1.897192732054405) q[4];
cx q[2],q[4];
ry(0.564014950231474) q[4];
ry(-1.8147260592569685) q[6];
cx q[4],q[6];
ry(1.0397340900618746) q[4];
ry(1.189170690757746) q[6];
cx q[4],q[6];
ry(-2.3814038152319976) q[1];
ry(-1.864643280735438) q[3];
cx q[1],q[3];
ry(-2.2746597509317) q[1];
ry(0.021772229723334924) q[3];
cx q[1],q[3];
ry(0.6976264678803048) q[3];
ry(-1.635893353793426) q[5];
cx q[3],q[5];
ry(-2.6674306139070656) q[3];
ry(3.069399242412973) q[5];
cx q[3],q[5];
ry(1.643158609360924) q[5];
ry(2.028014460645621) q[7];
cx q[5],q[7];
ry(1.0510768934768215) q[5];
ry(-0.5432319331809634) q[7];
cx q[5],q[7];
ry(2.4403472599867806) q[0];
ry(-2.9683369512667275) q[1];
cx q[0],q[1];
ry(-0.9736086923373435) q[0];
ry(1.51378960645309) q[1];
cx q[0],q[1];
ry(-2.3497643550883596) q[2];
ry(-1.3247632730829306) q[3];
cx q[2],q[3];
ry(-2.351476560950441) q[2];
ry(0.05448144376454777) q[3];
cx q[2],q[3];
ry(0.653166772243507) q[4];
ry(0.4342344945239391) q[5];
cx q[4],q[5];
ry(-0.35106102427706615) q[4];
ry(1.7249251172432327) q[5];
cx q[4],q[5];
ry(1.0262801581043792) q[6];
ry(-1.2036159459719036) q[7];
cx q[6],q[7];
ry(-0.5716545814148128) q[6];
ry(1.3301238983105304) q[7];
cx q[6],q[7];
ry(1.4683886239769093) q[0];
ry(-0.610061537737864) q[2];
cx q[0],q[2];
ry(-1.3306064252438263) q[0];
ry(2.077449843911939) q[2];
cx q[0],q[2];
ry(0.8543486876155026) q[2];
ry(-2.530646480072659) q[4];
cx q[2],q[4];
ry(0.9085892034046319) q[2];
ry(-0.5229407690564836) q[4];
cx q[2],q[4];
ry(0.514895462189681) q[4];
ry(0.5510132797424682) q[6];
cx q[4],q[6];
ry(2.7830298485588187) q[4];
ry(-2.9330029627816936) q[6];
cx q[4],q[6];
ry(3.0511562543705493) q[1];
ry(-0.49912568929008655) q[3];
cx q[1],q[3];
ry(-1.705266944198761) q[1];
ry(2.7960556139547794) q[3];
cx q[1],q[3];
ry(-3.020583041652436) q[3];
ry(0.7006193817558869) q[5];
cx q[3],q[5];
ry(-2.0219351742037546) q[3];
ry(-0.8216977026362698) q[5];
cx q[3],q[5];
ry(1.414687907690518) q[5];
ry(1.4502365439918106) q[7];
cx q[5],q[7];
ry(-0.4106023816863757) q[5];
ry(-1.3796951439687897) q[7];
cx q[5],q[7];
ry(-2.056295190583273) q[0];
ry(-3.0154072788207285) q[1];
cx q[0],q[1];
ry(-2.3552187272551324) q[0];
ry(-1.7802825813908552) q[1];
cx q[0],q[1];
ry(-1.366075791012788) q[2];
ry(3.0380699543852163) q[3];
cx q[2],q[3];
ry(0.9069984238742235) q[2];
ry(-0.600977491565434) q[3];
cx q[2],q[3];
ry(1.7359277954208467) q[4];
ry(-0.7868087949293244) q[5];
cx q[4],q[5];
ry(2.7672019132654553) q[4];
ry(1.610402047346488) q[5];
cx q[4],q[5];
ry(1.5467426882286364) q[6];
ry(1.3133562810325365) q[7];
cx q[6],q[7];
ry(-1.6822257257646855) q[6];
ry(-0.15639891956622592) q[7];
cx q[6],q[7];
ry(2.1634615123094303) q[0];
ry(2.290337376661701) q[2];
cx q[0],q[2];
ry(-1.7194586658478546) q[0];
ry(-1.2894140547473576) q[2];
cx q[0],q[2];
ry(0.7689885697641426) q[2];
ry(-1.4387997580819059) q[4];
cx q[2],q[4];
ry(1.8976902465550944) q[2];
ry(-2.517831558324866) q[4];
cx q[2],q[4];
ry(-2.720384321996348) q[4];
ry(-2.503681107010717) q[6];
cx q[4],q[6];
ry(-2.5658631809925114) q[4];
ry(-0.5542775713063047) q[6];
cx q[4],q[6];
ry(0.3697833420292209) q[1];
ry(-1.3081242428256425) q[3];
cx q[1],q[3];
ry(2.321962545510211) q[1];
ry(-2.270516178020242) q[3];
cx q[1],q[3];
ry(0.39868210144959976) q[3];
ry(-2.659127479128036) q[5];
cx q[3],q[5];
ry(-3.119502286172165) q[3];
ry(2.456429430856949) q[5];
cx q[3],q[5];
ry(-0.1345400819526237) q[5];
ry(0.9366344179580725) q[7];
cx q[5],q[7];
ry(2.973477582066346) q[5];
ry(-2.6926585399524683) q[7];
cx q[5],q[7];
ry(-3.007127741612963) q[0];
ry(-1.5927535187460284) q[1];
cx q[0],q[1];
ry(-0.7262696087760281) q[0];
ry(2.885885357037684) q[1];
cx q[0],q[1];
ry(0.500001561647899) q[2];
ry(1.3195695667395804) q[3];
cx q[2],q[3];
ry(2.288322137862137) q[2];
ry(3.096538942730987) q[3];
cx q[2],q[3];
ry(0.44569475440351614) q[4];
ry(2.4455059176731946) q[5];
cx q[4],q[5];
ry(0.6621622133545264) q[4];
ry(-2.0660992387759674) q[5];
cx q[4],q[5];
ry(2.6671708533826424) q[6];
ry(0.6965139715509983) q[7];
cx q[6],q[7];
ry(2.0241658600299166) q[6];
ry(0.3966729512625724) q[7];
cx q[6],q[7];
ry(-2.134520018361) q[0];
ry(-1.5980539363465187) q[2];
cx q[0],q[2];
ry(0.05600663632999523) q[0];
ry(1.3080467112101086) q[2];
cx q[0],q[2];
ry(-2.8879534016271737) q[2];
ry(1.0377036447831038) q[4];
cx q[2],q[4];
ry(-1.6474298522025488) q[2];
ry(2.8616800806799567) q[4];
cx q[2],q[4];
ry(2.6686315624663957) q[4];
ry(1.0230807546236123) q[6];
cx q[4],q[6];
ry(-0.8679241841684773) q[4];
ry(-1.4489932485700057) q[6];
cx q[4],q[6];
ry(2.3216352771176947) q[1];
ry(-3.1357964333623167) q[3];
cx q[1],q[3];
ry(1.4696561263813854) q[1];
ry(-1.9161276873842779) q[3];
cx q[1],q[3];
ry(1.4372754895255584) q[3];
ry(-2.1617888105801475) q[5];
cx q[3],q[5];
ry(0.18354862353706847) q[3];
ry(0.732846345799918) q[5];
cx q[3],q[5];
ry(3.0786928559956914) q[5];
ry(1.0749960843626845) q[7];
cx q[5],q[7];
ry(2.5582486988707975) q[5];
ry(-0.09165270429748383) q[7];
cx q[5],q[7];
ry(-1.6565308864018955) q[0];
ry(-0.40027506972385485) q[1];
ry(1.888037175206207) q[2];
ry(-1.707805037932551) q[3];
ry(1.9389479507159146) q[4];
ry(-2.82456552913826) q[5];
ry(-2.466655566949835) q[6];
ry(-0.7840769348461896) q[7];