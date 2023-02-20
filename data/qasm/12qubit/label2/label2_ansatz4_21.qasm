OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-3.1090257382547897) q[0];
rz(-2.7703867537410467) q[0];
ry(3.0974073335438295) q[1];
rz(1.4835543333173664) q[1];
ry(-1.8266384391570232) q[2];
rz(-0.2595104970313402) q[2];
ry(-1.4311151633458552) q[3];
rz(0.03203023677013661) q[3];
ry(1.5708185381629076) q[4];
rz(2.968811863813885) q[4];
ry(1.5692462352984267) q[5];
rz(0.33542596817290615) q[5];
ry(-1.5145210784365988) q[6];
rz(1.517747306782363) q[6];
ry(-0.13456447091526721) q[7];
rz(-0.21762895146053562) q[7];
ry(-3.121469613647398) q[8];
rz(-2.821777042184956) q[8];
ry(-0.004828083720088938) q[9];
rz(0.19597647887537928) q[9];
ry(1.2922342848004902) q[10];
rz(0.9187922631918395) q[10];
ry(2.1770996204034607) q[11];
rz(-2.600711845196191) q[11];
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
ry(2.6905448893471187) q[0];
rz(3.073874958240445) q[0];
ry(-2.325846770027114) q[1];
rz(0.20241670504376788) q[1];
ry(1.0190481705315706) q[2];
rz(-2.1694717872051923) q[2];
ry(-2.3279624266879195) q[3];
rz(-0.20743071703604346) q[3];
ry(-0.0018042278371678863) q[4];
rz(-1.3981598562297926) q[4];
ry(-0.0014777818340672155) q[5];
rz(1.228166240576308) q[5];
ry(0.009321836142841988) q[6];
rz(-0.7908148012923816) q[6];
ry(0.0002933681375116848) q[7];
rz(0.32108003463649504) q[7];
ry(-0.002726360855077381) q[8];
rz(0.725140602609383) q[8];
ry(-3.136999625436928) q[9];
rz(2.2019013037362054) q[9];
ry(0.6680019905853744) q[10];
rz(-0.47463031470471784) q[10];
ry(-1.2622142940891936) q[11];
rz(0.42055190182640695) q[11];
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
ry(0.18285424065012992) q[0];
rz(-1.9843379061463668) q[0];
ry(0.6796512234308689) q[1];
rz(2.1594388230030663) q[1];
ry(0.9313958557032969) q[2];
rz(-0.7986712203219625) q[2];
ry(1.888687018035709) q[3];
rz(-0.5771649416961209) q[3];
ry(1.4733245100385106) q[4];
rz(0.6308176917417603) q[4];
ry(-3.0374916603355544) q[5];
rz(1.5081537039111268) q[5];
ry(0.6773524773569902) q[6];
rz(-0.5894516736834341) q[6];
ry(-1.422739849043111) q[7];
rz(-2.8594038570941294) q[7];
ry(-1.6622357566185855) q[8];
rz(1.049625140501148) q[8];
ry(1.5711741997617628) q[9];
rz(1.4017215032339783) q[9];
ry(-0.7261469503494737) q[10];
rz(0.683228942208932) q[10];
ry(-1.66569934069733) q[11];
rz(1.3959312778006585) q[11];
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
ry(-2.9306461435865834) q[0];
rz(-2.5854025629308883) q[0];
ry(-0.3763787951325277) q[1];
rz(-0.4268652505534068) q[1];
ry(1.1714947801754036) q[2];
rz(-0.32400482091935157) q[2];
ry(-0.7526904530092412) q[3];
rz(-2.451170424221721) q[3];
ry(-3.1410744424273824) q[4];
rz(-1.6771195022566507) q[4];
ry(-3.1414599376552492) q[5];
rz(2.788879802953157) q[5];
ry(-0.0009058636016295685) q[6];
rz(0.38213315139636883) q[6];
ry(0.0004846701271328137) q[7];
rz(2.608095732195538) q[7];
ry(-0.0009964058700775089) q[8];
rz(1.3612495688017816) q[8];
ry(-3.134526605873224) q[9];
rz(1.8914398530699357) q[9];
ry(0.5932568115512049) q[10];
rz(2.1119672317520166) q[10];
ry(0.9277831984280729) q[11];
rz(2.4351208691897237) q[11];
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
ry(-0.43357917830281545) q[0];
rz(-3.078680731373666) q[0];
ry(-0.6197711143531168) q[1];
rz(1.3996432041909765) q[1];
ry(-2.416192697714039) q[2];
rz(-1.5812347176132027) q[2];
ry(-0.3202160042950658) q[3];
rz(-2.6791031744587763) q[3];
ry(-0.00017432060422617468) q[4];
rz(2.764354423286558) q[4];
ry(3.140694945012372) q[5];
rz(1.315043170533855) q[5];
ry(1.7080860403604587) q[6];
rz(-2.469428804008428) q[6];
ry(2.7697491349355206) q[7];
rz(0.8564060880474597) q[7];
ry(-1.8976131630046569) q[8];
rz(-2.454561953853655) q[8];
ry(2.3443365201807773) q[9];
rz(-0.6182371273896262) q[9];
ry(1.3294211112915761) q[10];
rz(2.540223226645017) q[10];
ry(-2.5921139576040044) q[11];
rz(-2.6662344326807474) q[11];
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
ry(-1.214187458758313) q[0];
rz(-2.590027807594387) q[0];
ry(-1.2397476600203197) q[1];
rz(-1.2023300303446234) q[1];
ry(-1.5667642317192467) q[2];
rz(-0.7278561010584907) q[2];
ry(-1.548308005398577) q[3];
rz(3.1336402104393883) q[3];
ry(3.1414008612389166) q[4];
rz(0.352709731477649) q[4];
ry(3.140994765963413) q[5];
rz(-3.09654981964547) q[5];
ry(-0.00034217042193337126) q[6];
rz(-2.553404735535945) q[6];
ry(3.1413950319405037) q[7];
rz(2.9368582838937654) q[7];
ry(2.844569847271771) q[8];
rz(-0.772212386497605) q[8];
ry(-2.8275731723166193) q[9];
rz(-2.2794517372891017) q[9];
ry(2.5789598977947756) q[10];
rz(3.0283855846663337) q[10];
ry(-2.085248364183365) q[11];
rz(0.7304312296816731) q[11];
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
ry(0.184573470027658) q[0];
rz(2.3669016778672476) q[0];
ry(1.9351294271020856) q[1];
rz(-2.2824619108196167) q[1];
ry(0.9982589092297387) q[2];
rz(-1.2762940918975942) q[2];
ry(-1.9648718583374167) q[3];
rz(2.454810820788715) q[3];
ry(-3.1150543905247496) q[4];
rz(1.4705574330706899) q[4];
ry(1.8476076086762874) q[5];
rz(-1.566900096608491) q[5];
ry(-0.9544880365566489) q[6];
rz(1.1511292978390788) q[6];
ry(2.441092049527694) q[7];
rz(2.361856247378268) q[7];
ry(-3.1323726166722063) q[8];
rz(0.17039139422242577) q[8];
ry(-0.009578826829968724) q[9];
rz(-1.9773297135146728) q[9];
ry(-3.1159601867729063) q[10];
rz(2.5221779403880054) q[10];
ry(0.015147183318033441) q[11];
rz(-1.419789560871256) q[11];
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
ry(1.2607230987692217) q[0];
rz(-2.165491205409457) q[0];
ry(0.5276356706498861) q[1];
rz(-2.5173900603392703) q[1];
ry(2.5335353755695547) q[2];
rz(-2.287541586885234) q[2];
ry(-1.964024274188843) q[3];
rz(0.8389071450595497) q[3];
ry(-1.7470238021890314) q[4];
rz(-0.8611979605373157) q[4];
ry(-1.3948220424284399) q[5];
rz(-0.3077869528951376) q[5];
ry(3.137618200401949) q[6];
rz(-2.658706215246056) q[6];
ry(0.007299319306251383) q[7];
rz(-2.781465744888529) q[7];
ry(2.6562343661099224) q[8];
rz(1.3142018493808734) q[8];
ry(-2.6941469931767923) q[9];
rz(0.39476916216203367) q[9];
ry(0.6449772044277307) q[10];
rz(1.839116641198601) q[10];
ry(-0.25420875921003017) q[11];
rz(0.5811577811193329) q[11];
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
ry(-0.7543450268223377) q[0];
rz(2.338750795904295) q[0];
ry(1.4617229369646338) q[1];
rz(-1.5880711377930645) q[1];
ry(0.9572326132038976) q[2];
rz(-1.9226078171605754) q[2];
ry(0.5879277349352909) q[3];
rz(-1.5885706680007432) q[3];
ry(-3.1393554807194963) q[4];
rz(0.5913899787312571) q[4];
ry(-3.1399715426458914) q[5];
rz(-1.832623831341538) q[5];
ry(1.6702651580243908) q[6];
rz(1.079073607155367) q[6];
ry(-1.5234107647274682) q[7];
rz(1.9945810069544907) q[7];
ry(3.1243800144220537) q[8];
rz(-0.36839130460855773) q[8];
ry(-3.057198835399959) q[9];
rz(-1.437728638939471) q[9];
ry(0.8470459795447569) q[10];
rz(2.2391483446697036) q[10];
ry(1.2451500432980191) q[11];
rz(0.41989381450722724) q[11];
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
ry(-1.6517469574135353) q[0];
rz(2.006149872731357) q[0];
ry(0.8138717212659466) q[1];
rz(1.94863471833896) q[1];
ry(1.2943511784513537) q[2];
rz(-1.204787543133467) q[2];
ry(-2.3910634082186566) q[3];
rz(-0.5416717350294106) q[3];
ry(0.0010197344516917894) q[4];
rz(1.673498021632878) q[4];
ry(3.138677164010601) q[5];
rz(-1.5497631129140979) q[5];
ry(-0.0025458700803546344) q[6];
rz(-0.7603297149328635) q[6];
ry(-0.006604069717798033) q[7];
rz(-0.2290308394082565) q[7];
ry(-3.138890172964642) q[8];
rz(0.1323695171728927) q[8];
ry(-3.1398907876361943) q[9];
rz(-0.5266377846180427) q[9];
ry(2.8747220656157775) q[10];
rz(3.077230153924642) q[10];
ry(-2.583616127278459) q[11];
rz(-0.5075148397747737) q[11];
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
ry(3.13274735675117) q[0];
rz(1.7896105140645873) q[0];
ry(0.29114802988495914) q[1];
rz(-0.33663004705685395) q[1];
ry(2.9460184216007623) q[2];
rz(-2.7721776877814346) q[2];
ry(-1.1189929120960125) q[3];
rz(1.1237754195035043) q[3];
ry(1.5962720979020129) q[4];
rz(-1.9673615229531611) q[4];
ry(-1.5546751927657434) q[5];
rz(-3.0939758172889245) q[5];
ry(0.28160146290622273) q[6];
rz(2.9064331022593426) q[6];
ry(-0.951962590192631) q[7];
rz(-1.5798826559762293) q[7];
ry(-2.499808946376929) q[8];
rz(-1.8933502817386305) q[8];
ry(-0.6353091298735762) q[9];
rz(1.3458019932423275) q[9];
ry(-1.7989322217282329) q[10];
rz(0.568045347068848) q[10];
ry(-3.1101926767952017) q[11];
rz(-2.266705026596238) q[11];
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
ry(-0.9260747570465441) q[0];
rz(1.6819860780554867) q[0];
ry(1.9421898715252468) q[1];
rz(1.580916524642908) q[1];
ry(1.3831492521996518) q[2];
rz(-0.17298953362800873) q[2];
ry(2.508335906715798) q[3];
rz(-2.053620328035279) q[3];
ry(-0.005217269665443781) q[4];
rz(-0.14862209552625574) q[4];
ry(0.010973010235477242) q[5];
rz(2.5443919545012044) q[5];
ry(-1.633080227768767) q[6];
rz(0.05063872440370787) q[6];
ry(-1.3100340936319201) q[7];
rz(2.3574141320460336) q[7];
ry(2.9328378658211745) q[8];
rz(1.1942757651454645) q[8];
ry(-0.2166601806159072) q[9];
rz(-2.058189468016541) q[9];
ry(-2.0025844909482258) q[10];
rz(-0.09944598470957722) q[10];
ry(-2.09987935218386) q[11];
rz(-2.728855685973879) q[11];
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
ry(-1.0522264956988359) q[0];
rz(0.5911565325457415) q[0];
ry(2.039812945170274) q[1];
rz(0.47084376907704056) q[1];
ry(3.136553852404898) q[2];
rz(-0.20955877651854954) q[2];
ry(-3.130191093746922) q[3];
rz(2.663443294416316) q[3];
ry(1.2160521168928158) q[4];
rz(2.732638205380419) q[4];
ry(1.9222915682886486) q[5];
rz(-0.4717463554511812) q[5];
ry(1.56748993649829) q[6];
rz(-1.1010840773959245) q[6];
ry(0.5414586768916789) q[7];
rz(1.7620330254809193) q[7];
ry(-2.894487863253034) q[8];
rz(3.0304851771092247) q[8];
ry(-0.718052675881462) q[9];
rz(2.085999875317921) q[9];
ry(2.2642657782030158) q[10];
rz(0.9629487582633502) q[10];
ry(-0.5990468947524005) q[11];
rz(1.11915315417107) q[11];
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
ry(-1.3759722751309047) q[0];
rz(2.3964672638970512) q[0];
ry(-1.9766784825529418) q[1];
rz(-1.4496390383704616) q[1];
ry(-1.61460197516191) q[2];
rz(2.9992452379700847) q[2];
ry(1.6282960417963395) q[3];
rz(-3.0452337355519927) q[3];
ry(2.521630460661245) q[4];
rz(0.02862796860252459) q[4];
ry(-0.6346509680079884) q[5];
rz(1.6792592851101318) q[5];
ry(3.1242427141393327) q[6];
rz(-1.9133912317534965) q[6];
ry(-3.129313774813656) q[7];
rz(-1.224627179217948) q[7];
ry(3.1034420605712136) q[8];
rz(0.10306925036352155) q[8];
ry(-0.00376526509411601) q[9];
rz(-1.0859001440256553) q[9];
ry(-2.1789090431635803) q[10];
rz(-0.3498046662936325) q[10];
ry(-2.1046538870755596) q[11];
rz(2.5571575873899466) q[11];
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
ry(-2.7896573287273805) q[0];
rz(2.725528114906761) q[0];
ry(-1.8023945664898757) q[1];
rz(0.8033251418233763) q[1];
ry(2.1324684410186663) q[2];
rz(-0.5148641423185523) q[2];
ry(1.4847271969547278) q[3];
rz(-1.7814044629432777) q[3];
ry(2.792704235920291) q[4];
rz(1.5068727310642622) q[4];
ry(-0.12325330812689624) q[5];
rz(3.0158323646253287) q[5];
ry(-2.9187179501576477) q[6];
rz(2.1951276429731754) q[6];
ry(-0.3192579882617464) q[7];
rz(2.1817272640075664) q[7];
ry(2.1672388097511215) q[8];
rz(1.3347247761411818) q[8];
ry(-2.5539100299215822) q[9];
rz(1.0658040646878009) q[9];
ry(-2.0915983556653694) q[10];
rz(1.8901340754869924) q[10];
ry(-2.577098377639269) q[11];
rz(-2.3337541423482504) q[11];
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
ry(-2.099239250343021) q[0];
rz(0.721291783074105) q[0];
ry(1.674636958640801) q[1];
rz(3.0316591664757753) q[1];
ry(-3.138036635322562) q[2];
rz(2.640375353061199) q[2];
ry(3.1399162660956343) q[3];
rz(-0.6599980087003542) q[3];
ry(-2.627217422237698) q[4];
rz(-1.7350620383088238) q[4];
ry(0.5213261531996994) q[5];
rz(-1.5836830959019181) q[5];
ry(0.23629524989208203) q[6];
rz(0.7673601997527495) q[6];
ry(-0.1951899473415022) q[7];
rz(-1.5288421659698441) q[7];
ry(-3.0821020004675614) q[8];
rz(0.6261531187837799) q[8];
ry(-3.0989427653907557) q[9];
rz(-1.9647213542175652) q[9];
ry(-2.0860032313406274) q[10];
rz(-1.2917336201342364) q[10];
ry(-1.6680809552046778) q[11];
rz(-0.36207264657537236) q[11];
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
ry(2.329546704024236) q[0];
rz(2.851782417116533) q[0];
ry(-1.7242251616182678) q[1];
rz(-3.1263526056610695) q[1];
ry(0.006634504923845978) q[2];
rz(1.6692899181694143) q[2];
ry(-3.1209808534265258) q[3];
rz(-1.9246377733639946) q[3];
ry(0.8091556965475419) q[4];
rz(-2.830374457358889) q[4];
ry(2.5592573369254534) q[5];
rz(-1.6559757498241983) q[5];
ry(-0.15599788881256593) q[6];
rz(1.7758979467785805) q[6];
ry(3.009682773586729) q[7];
rz(-1.1395260311856472) q[7];
ry(-2.9883258491125764) q[8];
rz(-0.14152649604755857) q[8];
ry(0.1624349549706598) q[9];
rz(-1.9644443824504465) q[9];
ry(0.3687811888957313) q[10];
rz(0.6099387895606779) q[10];
ry(2.7822263520930197) q[11];
rz(-1.9668833377462902) q[11];
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
ry(-1.419004268939984) q[0];
rz(-1.8459534691339354) q[0];
ry(-0.16506722011344888) q[1];
rz(1.034308774086513) q[1];
ry(0.005342011446712389) q[2];
rz(-0.9074492836636052) q[2];
ry(-0.02102578277534446) q[3];
rz(2.3819470992371947) q[3];
ry(0.03801517978542979) q[4];
rz(-0.49617837753360616) q[4];
ry(-0.0007062397322501468) q[5];
rz(2.8288370475695013) q[5];
ry(0.03528119949473041) q[6];
rz(-1.5924133946517776) q[6];
ry(3.093138556338376) q[7];
rz(-0.7697591152810263) q[7];
ry(-3.131973266783592) q[8];
rz(-2.8366002280554303) q[8];
ry(-0.08276586157519404) q[9];
rz(-1.8361649921314542) q[9];
ry(2.854497398860102) q[10];
rz(2.409102583767492) q[10];
ry(-2.4892714591815013) q[11];
rz(-0.5959895711647575) q[11];
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
ry(2.0200816296051034) q[0];
rz(-0.8917819649302601) q[0];
ry(2.0636330442656656) q[1];
rz(-0.9150223085765026) q[1];
ry(-0.2675772116944364) q[2];
rz(-0.7745653236815331) q[2];
ry(-2.749628434514272) q[3];
rz(2.2872048771418467) q[3];
ry(-1.7123102536865071) q[4];
rz(1.4745234454260299) q[4];
ry(3.0432928186443635) q[5];
rz(-2.794981949007173) q[5];
ry(-2.318415064484532) q[6];
rz(-1.4900184929322202) q[6];
ry(-0.8216213367008125) q[7];
rz(-1.3588740745227734) q[7];
ry(-0.8034102781340021) q[8];
rz(1.228289063930867) q[8];
ry(-2.4430049287454283) q[9];
rz(-2.8172779505107695) q[9];
ry(0.2992439107613074) q[10];
rz(1.6859168474561583) q[10];
ry(-2.109315362605469) q[11];
rz(1.4586080686678886) q[11];
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
ry(-1.3550820885367258) q[0];
rz(0.225744218769198) q[0];
ry(-1.3759684294522676) q[1];
rz(-2.8274941927617325) q[1];
ry(-3.079066960807247) q[2];
rz(-0.08442154638106265) q[2];
ry(-3.033025825541883) q[3];
rz(-2.7290319007880317) q[3];
ry(-3.1241791001257875) q[4];
rz(2.773254719425552) q[4];
ry(3.1095686409935754) q[5];
rz(-1.086461787486554) q[5];
ry(-2.715239434006586) q[6];
rz(1.9052096132134537) q[6];
ry(0.3510450037810866) q[7];
rz(-0.7071034739826327) q[7];
ry(0.6198376306210773) q[8];
rz(-1.013719076101708) q[8];
ry(0.4260288907285967) q[9];
rz(1.9875013913129145) q[9];
ry(-0.6321492962220131) q[10];
rz(-0.9053173300225854) q[10];
ry(2.9439411054627844) q[11];
rz(-2.069590079621298) q[11];
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
ry(2.3293205961051133) q[0];
rz(-0.48750405825400733) q[0];
ry(0.8938991869719454) q[1];
rz(-2.740926772591002) q[1];
ry(-0.7444419841829042) q[2];
rz(2.217057271642543) q[2];
ry(1.959721845248847) q[3];
rz(-0.06153987228010749) q[3];
ry(-3.0447277937999595) q[4];
rz(-0.1276328651502503) q[4];
ry(0.01963316046939756) q[5];
rz(-1.2873438116802154) q[5];
ry(-0.030855095573286032) q[6];
rz(0.8531730157542325) q[6];
ry(0.05895691702802974) q[7];
rz(-0.9277442912837515) q[7];
ry(0.011872030147540383) q[8];
rz(1.7118626425465175) q[8];
ry(-0.0135836772657294) q[9];
rz(0.9170709769254543) q[9];
ry(3.0058044219084397) q[10];
rz(1.2343798662782781) q[10];
ry(-0.09843750737439272) q[11];
rz(-1.6143050525743883) q[11];
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
ry(0.06834173361941076) q[0];
rz(2.7383330396620056) q[0];
ry(-0.06877654715743876) q[1];
rz(2.728598189971893) q[1];
ry(-3.090374947813231) q[2];
rz(-0.7577469144828533) q[2];
ry(0.0339401148202919) q[3];
rz(-1.2731204307315824) q[3];
ry(2.216449308154769) q[4];
rz(-1.1838549079038092) q[4];
ry(2.3445813873462504) q[5];
rz(-1.1265812516155576) q[5];
ry(-3.104961507905912) q[6];
rz(0.4525496997495635) q[6];
ry(2.5662851598237943) q[7];
rz(0.5105717203196173) q[7];
ry(0.4583398862315766) q[8];
rz(2.452307543566248) q[8];
ry(0.04022854641752093) q[9];
rz(-0.5348081397537375) q[9];
ry(-2.3160882904994167) q[10];
rz(-1.3485571748215621) q[10];
ry(2.2028778132914226) q[11];
rz(3.1338552162150184) q[11];
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
ry(-2.7278580785804376) q[0];
rz(1.7956997622217123) q[0];
ry(0.9802081742080025) q[1];
rz(-2.063619105724703) q[1];
ry(-0.2862910499777367) q[2];
rz(1.4328881336649943) q[2];
ry(2.8924996465959754) q[3];
rz(-2.694279442659392) q[3];
ry(3.1376258833887363) q[4];
rz(-2.9291294485624486) q[4];
ry(3.124492420506473) q[5];
rz(-2.8894488428042497) q[5];
ry(1.6841333320605123) q[6];
rz(-0.16865491104587974) q[6];
ry(1.6756978284769344) q[7];
rz(0.17594531885303688) q[7];
ry(0.038003101212614965) q[8];
rz(0.9692527499476888) q[8];
ry(-0.017915277263192666) q[9];
rz(-1.112028475636515) q[9];
ry(-1.9613462042380458) q[10];
rz(-2.4916567703800463) q[10];
ry(1.2922767296886102) q[11];
rz(-0.8464843963558729) q[11];
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
ry(-0.0017299327710151857) q[0];
rz(-1.970814500287024) q[0];
ry(0.009478140369784173) q[1];
rz(2.9670501456458456) q[1];
ry(0.005299866291426092) q[2];
rz(-0.02589954181470677) q[2];
ry(-0.009460722462597104) q[3];
rz(1.942894203962271) q[3];
ry(-1.60121999963732) q[4];
rz(0.6476566500474634) q[4];
ry(1.5407030297249937) q[5];
rz(2.6141512626681878) q[5];
ry(-1.6991162307654322) q[6];
rz(2.3822795334399105) q[6];
ry(-1.636117388381413) q[7];
rz(-2.0515424446704484) q[7];
ry(3.141544766089645) q[8];
rz(-2.889186091951762) q[8];
ry(-0.00041518912617899417) q[9];
rz(0.28863722419689597) q[9];
ry(-0.981852338202498) q[10];
rz(0.7708229239796046) q[10];
ry(0.9227749747525379) q[11];
rz(-0.6999311483419568) q[11];
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
ry(1.1981165494065715) q[0];
rz(-0.0715304095941125) q[0];
ry(0.9725215571974086) q[1];
rz(-0.8096673175292759) q[1];
ry(3.022710601892463) q[2];
rz(2.5017492520227527) q[2];
ry(-0.19723706491734921) q[3];
rz(1.7389905744049805) q[3];
ry(-1.6713908137806217) q[4];
rz(2.711298113234616) q[4];
ry(1.4980716658086024) q[5];
rz(2.6926606613646817) q[5];
ry(0.2161023648397728) q[6];
rz(1.3100412726955148) q[6];
ry(0.09525164955960541) q[7];
rz(1.6844905921086875) q[7];
ry(-1.7012741414547214) q[8];
rz(0.7727288343364922) q[8];
ry(-1.3475370196763432) q[9];
rz(-2.2991989644542556) q[9];
ry(1.1351647384684775) q[10];
rz(-1.7343565701988195) q[10];
ry(-1.2879388320029361) q[11];
rz(-2.4158846120287154) q[11];