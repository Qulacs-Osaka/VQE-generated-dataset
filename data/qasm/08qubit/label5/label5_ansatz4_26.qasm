OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.1832643023357883) q[0];
rz(-0.6010058268096863) q[0];
ry(-3.0405874696517783) q[1];
rz(2.675628150726597) q[1];
ry(-0.1593495934969001) q[2];
rz(-1.5230535487763028) q[2];
ry(1.7887841213090703) q[3];
rz(2.4142503434482654) q[3];
ry(2.2366514966587223) q[4];
rz(-1.0956315148118412) q[4];
ry(1.3812854325194035) q[5];
rz(2.8958012448252637) q[5];
ry(-0.5872286097502826) q[6];
rz(-3.0679526823232455) q[6];
ry(-1.6634387398739454) q[7];
rz(-2.4951695295797713) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.721230107295076) q[0];
rz(1.4051758083618198) q[0];
ry(0.274785499115759) q[1];
rz(-2.0642994962187755) q[1];
ry(-0.23612508584233804) q[2];
rz(-2.80371954646374) q[2];
ry(1.517166475287298) q[3];
rz(-1.4460128940858261) q[3];
ry(-2.1599219752919367) q[4];
rz(-2.1055006746176104) q[4];
ry(2.463951262710052) q[5];
rz(-1.5275056946571042) q[5];
ry(0.23681675233609795) q[6];
rz(1.00196586165238) q[6];
ry(0.31060978409573764) q[7];
rz(1.5089272480453157) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.244287935065512) q[0];
rz(-1.72010744689437) q[0];
ry(-2.15280070397532) q[1];
rz(-2.450604525116395) q[1];
ry(2.943279014704652) q[2];
rz(-1.0013474738102301) q[2];
ry(-0.3614883425020139) q[3];
rz(1.3306528153308887) q[3];
ry(-3.0486384130971036) q[4];
rz(1.5653583984991952) q[4];
ry(-1.2268540003967943) q[5];
rz(-2.470468346333469) q[5];
ry(-1.4873178583096776) q[6];
rz(-2.9994227558197686) q[6];
ry(-2.8978929151695527) q[7];
rz(0.35216687215836723) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.39950424812436847) q[0];
rz(1.7289641051972566) q[0];
ry(-2.6717997696032993) q[1];
rz(0.3444012049566241) q[1];
ry(0.7906589670193069) q[2];
rz(-2.0104153958969952) q[2];
ry(-1.0608400649188487) q[3];
rz(-1.844751217913637) q[3];
ry(-2.4282431911944244) q[4];
rz(-2.3097978652026057) q[4];
ry(-2.209507398431407) q[5];
rz(-2.352580370985584) q[5];
ry(0.861296523945921) q[6];
rz(-0.07378084867400059) q[6];
ry(-1.476406922428058) q[7];
rz(-0.10018327778072532) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.9532588655517336) q[0];
rz(0.22055534386737818) q[0];
ry(0.1477452710727221) q[1];
rz(-2.775822242212249) q[1];
ry(-1.7189801663921844) q[2];
rz(0.9292115971647457) q[2];
ry(0.5497415635471743) q[3];
rz(-1.8842210026587907) q[3];
ry(1.5149061450362193) q[4];
rz(1.2161608925955125) q[4];
ry(-1.2276385012017412) q[5];
rz(-2.4380311475794265) q[5];
ry(1.9494015360544097) q[6];
rz(-1.4446817282997066) q[6];
ry(-0.9066892132842939) q[7];
rz(-0.9918624979760615) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.8759900386979456) q[0];
rz(1.5754965061277835) q[0];
ry(-0.637892061721109) q[1];
rz(-1.7710934796325997) q[1];
ry(1.0754350978631801) q[2];
rz(0.6072156425723776) q[2];
ry(-0.30984336138091) q[3];
rz(0.11906527483459682) q[3];
ry(-2.1941312927105248) q[4];
rz(-1.3164758694427383) q[4];
ry(2.6322148301604282) q[5];
rz(-0.8026550557212825) q[5];
ry(-0.30176308173822086) q[6];
rz(1.3033249912681772) q[6];
ry(0.6189133238604675) q[7];
rz(2.431013832802527) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.4378571583509867) q[0];
rz(3.0002612719196424) q[0];
ry(2.3333880987663265) q[1];
rz(-0.4278652058198329) q[1];
ry(-1.725429725302023) q[2];
rz(0.5592154175789021) q[2];
ry(1.1310647197999817) q[3];
rz(-2.4723305759750533) q[3];
ry(-2.243324470996458) q[4];
rz(2.5504235106560147) q[4];
ry(2.240229165640965) q[5];
rz(2.5524156540622434) q[5];
ry(-2.7658017399124684) q[6];
rz(0.5230664031813524) q[6];
ry(3.027928977318959) q[7];
rz(1.0793261502186011) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.827065100304371) q[0];
rz(-1.9521547861755886) q[0];
ry(-0.16624377508048988) q[1];
rz(-1.6482014727434584) q[1];
ry(0.94710495078005) q[2];
rz(-3.074011960436804) q[2];
ry(0.050777661369472156) q[3];
rz(2.072010233243386) q[3];
ry(-1.9617350793663764) q[4];
rz(3.0056633195458407) q[4];
ry(-2.060050796927359) q[5];
rz(0.3530093473698077) q[5];
ry(-1.2180507153845124) q[6];
rz(-2.69077445636258) q[6];
ry(-2.7699378858550987) q[7];
rz(0.40916815403984214) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.1336718589353474) q[0];
rz(0.5706260066121428) q[0];
ry(2.222283711446622) q[1];
rz(0.5165075729149484) q[1];
ry(2.796671322507168) q[2];
rz(2.3714388693133444) q[2];
ry(0.9304469384007943) q[3];
rz(0.4264757771364946) q[3];
ry(-0.42927245905323946) q[4];
rz(-2.9970324149459087) q[4];
ry(-2.1635434857872533) q[5];
rz(-0.09382513516106754) q[5];
ry(1.3744199523568514) q[6];
rz(-1.5088026093983595) q[6];
ry(-0.9831632341506618) q[7];
rz(-2.531326638523173) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.67622125168538) q[0];
rz(0.6022028669412807) q[0];
ry(1.9629349551010162) q[1];
rz(-1.4746075212208554) q[1];
ry(-1.566368865797754) q[2];
rz(2.598638957574408) q[2];
ry(-2.039052190076608) q[3];
rz(1.3826635919404953) q[3];
ry(-2.8863711072946066) q[4];
rz(-2.776065344744214) q[4];
ry(-1.3315573637690026) q[5];
rz(-0.635149943309055) q[5];
ry(1.3837386063543233) q[6];
rz(-0.8735864096397351) q[6];
ry(-1.4047175724907266) q[7];
rz(2.8712215617277037) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.4813651832396735) q[0];
rz(-2.6208515630171467) q[0];
ry(0.22413331241993095) q[1];
rz(-0.0341054775419069) q[1];
ry(-1.9925569697153185) q[2];
rz(-2.2433364239811815) q[2];
ry(0.6921284697007021) q[3];
rz(-1.312164760940001) q[3];
ry(1.1423911263925701) q[4];
rz(-1.0729647895680925) q[4];
ry(1.7405517368328214) q[5];
rz(1.9195101655476483) q[5];
ry(2.3878119306837164) q[6];
rz(-1.6009995621693438) q[6];
ry(1.942508044860788) q[7];
rz(-1.0142302159244236) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.05530696560133584) q[0];
rz(-3.073868452536517) q[0];
ry(2.4228278750945806) q[1];
rz(-2.8443746993579446) q[1];
ry(-2.128688673674845) q[2];
rz(-0.6318399899400828) q[2];
ry(-2.2918274980202575) q[3];
rz(-1.1013711812674403) q[3];
ry(0.24395989602582535) q[4];
rz(1.2228005696251951) q[4];
ry(1.1583312789498956) q[5];
rz(-3.1103212985114816) q[5];
ry(1.7653787857505483) q[6];
rz(1.2747402845989928) q[6];
ry(0.4015508919404951) q[7];
rz(1.9040254399917411) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.1470986713454228) q[0];
rz(-2.957669834398347) q[0];
ry(-2.5701817731041405) q[1];
rz(1.6899051629502067) q[1];
ry(-2.7361547782500546) q[2];
rz(0.07163206409653156) q[2];
ry(-1.3778622235511664) q[3];
rz(0.8404520611097906) q[3];
ry(2.0162947131806206) q[4];
rz(2.5778743804518713) q[4];
ry(-0.03398953495201018) q[5];
rz(0.43761292062348034) q[5];
ry(-1.3016932684979385) q[6];
rz(2.12449967703007) q[6];
ry(0.7297069893514684) q[7];
rz(-2.529533273174204) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.47783212186481716) q[0];
rz(1.9307519105556032) q[0];
ry(-0.41091820592172923) q[1];
rz(1.4898415128708418) q[1];
ry(-0.30546552618772965) q[2];
rz(-0.7124732186664168) q[2];
ry(-1.3113101955054385) q[3];
rz(-1.3236465807659124) q[3];
ry(-2.611730424156032) q[4];
rz(0.2569526765285044) q[4];
ry(2.2436217947562467) q[5];
rz(-1.193803443050613) q[5];
ry(2.330372448534692) q[6];
rz(-2.1865757622940003) q[6];
ry(0.7618657430011048) q[7];
rz(1.153558676103824) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.8118113867730035) q[0];
rz(1.7724736231536458) q[0];
ry(1.034045051610423) q[1];
rz(-1.9101830066922443) q[1];
ry(0.9706548487294153) q[2];
rz(0.6978029612104333) q[2];
ry(-2.979272143522874) q[3];
rz(-0.42921594258322643) q[3];
ry(-2.3850656378806927) q[4];
rz(0.6783962948215557) q[4];
ry(1.7047772890804662) q[5];
rz(-2.1737158743873355) q[5];
ry(1.8044867719138118) q[6];
rz(-1.5029868770302168) q[6];
ry(-1.2924759369625174) q[7];
rz(1.5537585406114482) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.041837271408158) q[0];
rz(0.3215084054860906) q[0];
ry(1.6771967123260771) q[1];
rz(-0.10469174128200633) q[1];
ry(0.4889884218427678) q[2];
rz(-0.5211851764466395) q[2];
ry(-1.8126255325524578) q[3];
rz(1.6430769957376707) q[3];
ry(0.9808537696154866) q[4];
rz(1.5569083432846935) q[4];
ry(-3.073536024703657) q[5];
rz(-3.0424389683464548) q[5];
ry(-0.49253632859692154) q[6];
rz(1.4459834052346547) q[6];
ry(2.5354402137595233) q[7];
rz(1.8619732621953167) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.1701961104132304) q[0];
rz(1.790213623875936) q[0];
ry(-2.708567402888844) q[1];
rz(-0.8015556718366583) q[1];
ry(-1.182202680497495) q[2];
rz(2.5645558417025724) q[2];
ry(-1.801588403139652) q[3];
rz(1.9987069103394814) q[3];
ry(-0.3109330609136576) q[4];
rz(3.096246299607342) q[4];
ry(-0.5956037724804932) q[5];
rz(-2.993611208689028) q[5];
ry(-1.1139026652404675) q[6];
rz(0.9433916862388692) q[6];
ry(-3.0916106354004165) q[7];
rz(0.8207187109408256) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.7262608993618196) q[0];
rz(-2.1726203111394984) q[0];
ry(1.0162190012162153) q[1];
rz(0.15439986171168663) q[1];
ry(1.5345524892163844) q[2];
rz(-1.5732143852870237) q[2];
ry(0.047992194744262306) q[3];
rz(0.1614933145119641) q[3];
ry(-1.8709290729155041) q[4];
rz(-1.3494161894476226) q[4];
ry(1.3647919924070526) q[5];
rz(0.5797783564733061) q[5];
ry(1.6140607187249554) q[6];
rz(-0.14574357375411484) q[6];
ry(-1.9637082759850455) q[7];
rz(-0.8509708323444982) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.0324881560083417) q[0];
rz(-1.8929575921375923) q[0];
ry(-0.5093364602599628) q[1];
rz(0.6473609776706057) q[1];
ry(0.21755120172672268) q[2];
rz(2.6947219802378695) q[2];
ry(-2.0570355647968266) q[3];
rz(-1.3911320057339134) q[3];
ry(-3.0631079532701735) q[4];
rz(2.7453317052852846) q[4];
ry(-1.9989064943256487) q[5];
rz(0.17229754357903054) q[5];
ry(1.3590609837445122) q[6];
rz(0.5941671527946898) q[6];
ry(2.014356245073823) q[7];
rz(1.3279872725475936) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.6534384036787433) q[0];
rz(-2.388530929262156) q[0];
ry(-0.9578562706017824) q[1];
rz(0.5185231308789423) q[1];
ry(0.36383504183546567) q[2];
rz(-2.6955173499371954) q[2];
ry(-0.6584001846967045) q[3];
rz(-0.9707436159899987) q[3];
ry(0.6957207858640304) q[4];
rz(2.443932523190784) q[4];
ry(1.8195439516397427) q[5];
rz(-1.0410595536990852) q[5];
ry(-0.37031025991140815) q[6];
rz(1.9406069927911718) q[6];
ry(2.2554847943208802) q[7];
rz(2.027537028986743) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.494049454793074) q[0];
rz(-2.2601283860816483) q[0];
ry(2.115855264361283) q[1];
rz(-3.0378586908671537) q[1];
ry(2.464979749919702) q[2];
rz(-0.8590182317183261) q[2];
ry(-0.5128737637630213) q[3];
rz(-0.22173791426077777) q[3];
ry(1.9142782093130906) q[4];
rz(0.08520493268112031) q[4];
ry(2.951265124991934) q[5];
rz(-1.8867827947598217) q[5];
ry(3.00195301023179) q[6];
rz(-2.365653458391192) q[6];
ry(-1.3060142276665967) q[7];
rz(0.8767238027306504) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.3131502630437035) q[0];
rz(-0.3077056562870311) q[0];
ry(-2.759299937662903) q[1];
rz(-0.5867752324890197) q[1];
ry(-0.567119143836832) q[2];
rz(1.8677395040607578) q[2];
ry(1.0895018386587174) q[3];
rz(-1.2992804454394182) q[3];
ry(1.655701017810145) q[4];
rz(-2.636143737603843) q[4];
ry(-0.5872617968063821) q[5];
rz(-3.133678729141882) q[5];
ry(2.302733198991922) q[6];
rz(-0.5486113253842269) q[6];
ry(0.3538748318326947) q[7];
rz(-2.4675854359229596) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.9913684083376175) q[0];
rz(-2.0962815016576464) q[0];
ry(-1.3743134515162934) q[1];
rz(-3.059695157786173) q[1];
ry(-0.466046386896501) q[2];
rz(-1.6337359983649211) q[2];
ry(-2.4180978768780936) q[3];
rz(1.8336453574129061) q[3];
ry(2.2591691102425564) q[4];
rz(-2.7058960305898916) q[4];
ry(-2.237945961564625) q[5];
rz(0.46122818328633736) q[5];
ry(1.9894842106140942) q[6];
rz(-2.41571513186672) q[6];
ry(-2.6268380999083374) q[7];
rz(0.4012342455729021) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.007488416487921) q[0];
rz(0.7354635045787035) q[0];
ry(0.6332052704208346) q[1];
rz(3.0631119759396856) q[1];
ry(-0.061460200986249686) q[2];
rz(0.4229800161989199) q[2];
ry(-1.0051804944592169) q[3];
rz(-1.9345009972657476) q[3];
ry(2.7300690404974954) q[4];
rz(-1.045184696048978) q[4];
ry(-2.095505465390486) q[5];
rz(-2.927552197423835) q[5];
ry(-0.18619585130165195) q[6];
rz(2.9515838129264163) q[6];
ry(1.3053455854507068) q[7];
rz(-1.6624229756282773) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5507010208842784) q[0];
rz(0.7020456150706702) q[0];
ry(1.703024477605167) q[1];
rz(1.426732034247193) q[1];
ry(-2.4577104687807) q[2];
rz(-0.8840905167953597) q[2];
ry(2.5029595976165084) q[3];
rz(2.1329903736135307) q[3];
ry(1.2662323872967054) q[4];
rz(-2.6003531273569354) q[4];
ry(-2.2987610133292935) q[5];
rz(-1.3193554423476774) q[5];
ry(-0.48054905490269206) q[6];
rz(-3.011002009270914) q[6];
ry(-0.11892781668149889) q[7];
rz(-1.3120920435946009) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.223920688834096) q[0];
rz(-2.457078880987348) q[0];
ry(-2.2818963510750283) q[1];
rz(-1.3338225087672821) q[1];
ry(-2.2649664787474046) q[2];
rz(-0.7681112244198118) q[2];
ry(-1.1230099780243619) q[3];
rz(-1.3710098073908323) q[3];
ry(-0.3244228348044337) q[4];
rz(0.5490173067502848) q[4];
ry(-1.1629762651602524) q[5];
rz(-2.1943870933644756) q[5];
ry(-1.189504592962127) q[6];
rz(-1.553327957962887) q[6];
ry(2.3460520906237434) q[7];
rz(1.8709658096916137) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.9374549472816547) q[0];
rz(-1.1631742483980778) q[0];
ry(2.636043061242612) q[1];
rz(1.5435606065140632) q[1];
ry(-2.3724037980300903) q[2];
rz(1.701230237385741) q[2];
ry(-1.4214318615427493) q[3];
rz(0.23111673408700023) q[3];
ry(0.21217431277064433) q[4];
rz(2.625905854082742) q[4];
ry(2.0207954447279315) q[5];
rz(0.7536182082480103) q[5];
ry(-1.862476294344683) q[6];
rz(1.9706819441183772) q[6];
ry(0.2281947510918631) q[7];
rz(3.109491773000231) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.3462509253353385) q[0];
rz(-1.0007008663586348) q[0];
ry(-1.5079380061265315) q[1];
rz(-0.553218785365277) q[1];
ry(-1.6452327978080357) q[2];
rz(0.2044177625309764) q[2];
ry(-0.31510116709560076) q[3];
rz(-0.9774196096109296) q[3];
ry(-0.9535463224572542) q[4];
rz(1.0877931465279447) q[4];
ry(2.0378177222101375) q[5];
rz(-0.27626487951631074) q[5];
ry(2.73978721819162) q[6];
rz(-0.19523061060888922) q[6];
ry(0.3415307708228843) q[7];
rz(-2.044093587357497) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.113616139628906) q[0];
rz(-1.1150113741850323) q[0];
ry(-2.653923568538222) q[1];
rz(-1.2249712307443499) q[1];
ry(2.8035677392648206) q[2];
rz(-1.4378220662107601) q[2];
ry(2.3243574833375877) q[3];
rz(2.6365971312448977) q[3];
ry(-0.09760551393142425) q[4];
rz(2.126441683346493) q[4];
ry(-2.115583275248078) q[5];
rz(-0.8497508806810864) q[5];
ry(-2.56638859849205) q[6];
rz(-0.39925003943081805) q[6];
ry(1.2457824082013447) q[7];
rz(-2.427736450562589) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.7896201893935704) q[0];
rz(1.9194671141822317) q[0];
ry(2.1115076311716527) q[1];
rz(-0.4822896976925826) q[1];
ry(-3.056528307548846) q[2];
rz(-0.8017685839443458) q[2];
ry(-1.4199113082926225) q[3];
rz(1.832811262937584) q[3];
ry(-0.14957744958264296) q[4];
rz(-2.0965017178902103) q[4];
ry(-0.38566172068059534) q[5];
rz(-1.588184547187101) q[5];
ry(2.132299960940408) q[6];
rz(-2.3387493315498866) q[6];
ry(-1.5500244287690534) q[7];
rz(1.1558671679351251) q[7];