OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.1812800928612743) q[0];
rz(0.7324257280602479) q[0];
ry(-0.27192576824975806) q[1];
rz(-0.5536931684467588) q[1];
ry(2.7553138534645356) q[2];
rz(3.1074367707364083) q[2];
ry(2.386694467008133) q[3];
rz(-1.374064812192339) q[3];
ry(2.703895415605461) q[4];
rz(0.5961084715032854) q[4];
ry(-0.9539246905887032) q[5];
rz(-0.4720552028609672) q[5];
ry(0.1649230099317946) q[6];
rz(-2.615639552816292) q[6];
ry(2.8638621769795014) q[7];
rz(2.294819020689101) q[7];
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
ry(-2.7610783302990933) q[0];
rz(2.1631021641462898) q[0];
ry(-2.996657277520189) q[1];
rz(-0.04347464787410704) q[1];
ry(-1.253411277803667) q[2];
rz(2.5912281298882225) q[2];
ry(-0.988619026414951) q[3];
rz(2.086550817666037) q[3];
ry(1.488733873443062) q[4];
rz(2.5958500330150907) q[4];
ry(1.1687303490126013) q[5];
rz(-2.8547537909875413) q[5];
ry(2.468464910417166) q[6];
rz(-0.7009569197930468) q[6];
ry(-0.37812591223165304) q[7];
rz(1.6362835614122635) q[7];
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
ry(-0.8216678716311173) q[0];
rz(2.493238322665764) q[0];
ry(-1.8121892193782299) q[1];
rz(-2.2459915146023413) q[1];
ry(-1.8008417354883965) q[2];
rz(0.1857409629290824) q[2];
ry(-2.0973523239169873) q[3];
rz(0.35636914417506826) q[3];
ry(-1.7107988235138276) q[4];
rz(0.8868348603623718) q[4];
ry(1.8543850844113963) q[5];
rz(0.10425343996791771) q[5];
ry(2.69603954054603) q[6];
rz(-2.445884193945022) q[6];
ry(1.298503826058901) q[7];
rz(-0.40684312646736465) q[7];
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
ry(-1.8224221761516608) q[0];
rz(1.6162744439742065) q[0];
ry(0.478852079667113) q[1];
rz(2.2854565745184074) q[1];
ry(1.5644334001162643) q[2];
rz(-1.355720051350132) q[2];
ry(-0.8094680728848325) q[3];
rz(0.6198236038611958) q[3];
ry(0.22761880485930863) q[4];
rz(3.118590718979507) q[4];
ry(0.7744139822887579) q[5];
rz(2.546373727487135) q[5];
ry(2.0633921636516885) q[6];
rz(2.8566471075733593) q[6];
ry(-0.7380894308859229) q[7];
rz(2.4560475946206073) q[7];
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
ry(2.6982126127859494) q[0];
rz(-0.42962185062138286) q[0];
ry(-1.692259479987209) q[1];
rz(-2.336113179743468) q[1];
ry(1.1266947650021688) q[2];
rz(-1.1346401246035718) q[2];
ry(-2.0122500061479682) q[3];
rz(1.1563085174156136) q[3];
ry(-2.1039332756508395) q[4];
rz(2.5562022308744305) q[4];
ry(-2.0533880621886436) q[5];
rz(3.0641148989898577) q[5];
ry(-2.405275297283352) q[6];
rz(-1.3031085496323926) q[6];
ry(-0.9955786346147387) q[7];
rz(0.48583268277542047) q[7];
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
ry(-1.277724758292063) q[0];
rz(0.6258808021950433) q[0];
ry(-1.479912268505318) q[1];
rz(1.4774158440507479) q[1];
ry(-0.9295532363862695) q[2];
rz(0.7595058870804996) q[2];
ry(-2.5498987094057335) q[3];
rz(1.5179697960872893) q[3];
ry(-1.008734491111252) q[4];
rz(-0.8430436495402739) q[4];
ry(-1.4915035386354936) q[5];
rz(-2.4687338253006303) q[5];
ry(-2.27753537851521) q[6];
rz(-0.3562152411990525) q[6];
ry(-1.5820835864112306) q[7];
rz(-2.654361452281044) q[7];
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
ry(0.5864735039037079) q[0];
rz(0.8369328929433207) q[0];
ry(2.0108872576097054) q[1];
rz(0.37856843189936384) q[1];
ry(3.0943754850384155) q[2];
rz(1.891793694915047) q[2];
ry(1.2221890887033007) q[3];
rz(2.1538330731092135) q[3];
ry(1.4871251477183964) q[4];
rz(0.7504911668116496) q[4];
ry(-2.159991552986123) q[5];
rz(1.7206370304778373) q[5];
ry(-0.7846892776600569) q[6];
rz(0.7192780662916975) q[6];
ry(1.710842845495396) q[7];
rz(-2.3743216212683276) q[7];
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
ry(-2.5277557856090747) q[0];
rz(2.5856744246305152) q[0];
ry(-0.5907937253019191) q[1];
rz(-2.650605594198523) q[1];
ry(2.0336821435174866) q[2];
rz(-0.48080213749466627) q[2];
ry(2.488302780924248) q[3];
rz(-0.30440564916419177) q[3];
ry(1.875227907527293) q[4];
rz(-2.179476113879004) q[4];
ry(0.5748102658204699) q[5];
rz(-2.81859149132027) q[5];
ry(-0.7165244448464202) q[6];
rz(0.9722852111829826) q[6];
ry(1.6438676772919116) q[7];
rz(1.3508146723845922) q[7];
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
ry(-1.0300568689226264) q[0];
rz(-0.9144424172167809) q[0];
ry(2.5366227398988945) q[1];
rz(1.1269334367157409) q[1];
ry(2.3537828032472095) q[2];
rz(1.1376469078088656) q[2];
ry(0.21997679754357233) q[3];
rz(-1.8355788409142475) q[3];
ry(-0.44039485138367856) q[4];
rz(1.0955357890563118) q[4];
ry(1.7973346725319435) q[5];
rz(-1.7974903030425615) q[5];
ry(2.101464652833549) q[6];
rz(0.0586770406159518) q[6];
ry(-1.6335277238187462) q[7];
rz(-1.4869777652418765) q[7];
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
ry(-1.2694025976724) q[0];
rz(0.30038001258474106) q[0];
ry(0.8569530186193468) q[1];
rz(2.401895083236404) q[1];
ry(-0.4895866671574111) q[2];
rz(2.884549784385975) q[2];
ry(1.7229514940555584) q[3];
rz(1.3084422939730507) q[3];
ry(0.6397459302923992) q[4];
rz(-0.2260879919117181) q[4];
ry(-1.2604633226255824) q[5];
rz(1.9358334159980517) q[5];
ry(-1.4173635574584342) q[6];
rz(-0.5219410353673534) q[6];
ry(-1.5815356477562403) q[7];
rz(2.09454891806717) q[7];
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
ry(-0.17500460032564066) q[0];
rz(-2.468498974163807) q[0];
ry(3.075426194713202) q[1];
rz(-1.2586783358910703) q[1];
ry(0.6953581169396635) q[2];
rz(-0.2543904914467845) q[2];
ry(0.8502076821368574) q[3];
rz(2.856688237412258) q[3];
ry(2.883443260158516) q[4];
rz(-0.47053339601195093) q[4];
ry(0.6563072964976461) q[5];
rz(2.451352619915744) q[5];
ry(-1.704540562275244) q[6];
rz(2.338544176849841) q[6];
ry(-0.2841918452017811) q[7];
rz(2.738566432835102) q[7];
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
ry(0.42342019560356625) q[0];
rz(2.1870869426243846) q[0];
ry(1.5449947363102685) q[1];
rz(-1.2719314770340278) q[1];
ry(-1.400884857481671) q[2];
rz(2.352830185320159) q[2];
ry(-1.3062874376790448) q[3];
rz(-1.0057219748649642) q[3];
ry(-2.7440180218191763) q[4];
rz(0.6985137274652198) q[4];
ry(1.7240606328891221) q[5];
rz(1.5610292503313588) q[5];
ry(0.5663499188406824) q[6];
rz(2.363343676345259) q[6];
ry(-2.8504670996843275) q[7];
rz(0.23044804939533897) q[7];
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
ry(-1.7715752169522978) q[0];
rz(-2.320622847728238) q[0];
ry(1.8623982777613914) q[1];
rz(0.5180419072456042) q[1];
ry(1.4782277642251391) q[2];
rz(1.8331460410075506) q[2];
ry(0.47804728451179895) q[3];
rz(2.8179289056053767) q[3];
ry(-0.537531837046136) q[4];
rz(-2.6131325185838365) q[4];
ry(-1.991389044262644) q[5];
rz(2.7187729886735) q[5];
ry(0.558542557932098) q[6];
rz(1.633491468264502) q[6];
ry(3.020581589274539) q[7];
rz(0.544880534497945) q[7];
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
ry(2.1222293254522704) q[0];
rz(3.016764577655063) q[0];
ry(1.916268082424472) q[1];
rz(1.770304936347591) q[1];
ry(-1.741960077593431) q[2];
rz(-0.29724367655102146) q[2];
ry(2.6558482186886248) q[3];
rz(-2.5307316971551588) q[3];
ry(-2.199523852959696) q[4];
rz(-0.14266224770259214) q[4];
ry(1.0553646125155645) q[5];
rz(-1.1496995676129358) q[5];
ry(1.359324089296303) q[6];
rz(-2.997004012894444) q[6];
ry(-1.2296587909467445) q[7];
rz(2.688121879539503) q[7];
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
ry(2.945577919921273) q[0];
rz(-1.1288840231987338) q[0];
ry(-1.4002851099083466) q[1];
rz(-0.41370234970226977) q[1];
ry(2.641058108128138) q[2];
rz(-0.7770157236392475) q[2];
ry(-1.1545803033117146) q[3];
rz(1.1437110883854866) q[3];
ry(0.9725125094528195) q[4];
rz(1.79921638147997) q[4];
ry(-0.9899445568448288) q[5];
rz(2.0151113951917052) q[5];
ry(-1.3979518298765863) q[6];
rz(1.993241771763885) q[6];
ry(-2.013439133136073) q[7];
rz(3.0900042020478473) q[7];
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
ry(-1.3859171090532092) q[0];
rz(0.18200912236753553) q[0];
ry(-0.17505916547710199) q[1];
rz(-2.4008634862599507) q[1];
ry(-1.7806262079538795) q[2];
rz(2.897929400300995) q[2];
ry(1.9824498493852554) q[3];
rz(1.9991342373402414) q[3];
ry(-2.6626108573254457) q[4];
rz(-1.2161956967654892) q[4];
ry(2.0380987780496866) q[5];
rz(0.018966469955653764) q[5];
ry(1.4688598131300714) q[6];
rz(-2.8857055614738005) q[6];
ry(-2.9911678535417696) q[7];
rz(1.003183746031339) q[7];
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
ry(-1.1916843409068831) q[0];
rz(-1.6315576852358555) q[0];
ry(1.4790205353755068) q[1];
rz(-1.1560110330737166) q[1];
ry(0.7833172749573672) q[2];
rz(-0.8085281571750293) q[2];
ry(-2.3413187846796752) q[3];
rz(-1.3140556590601749) q[3];
ry(-0.1621078565188895) q[4];
rz(2.4665997607609387) q[4];
ry(-2.0165532661563996) q[5];
rz(-0.1656663002793302) q[5];
ry(1.1574504769915406) q[6];
rz(1.6867380768616398) q[6];
ry(2.9952608597627655) q[7];
rz(-1.8977472619832607) q[7];
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
ry(1.8200895434311848) q[0];
rz(0.9703572354509056) q[0];
ry(3.060727244813048) q[1];
rz(0.5040412645637541) q[1];
ry(-1.069275145003929) q[2];
rz(-0.7787580489153739) q[2];
ry(-1.1080679657204928) q[3];
rz(-2.2821039416233195) q[3];
ry(-1.8731228494157657) q[4];
rz(-1.3359622257322359) q[4];
ry(2.5273093004157374) q[5];
rz(-2.0767268524067313) q[5];
ry(-2.215900727529106) q[6];
rz(2.7131168573707583) q[6];
ry(0.719171006028545) q[7];
rz(2.2101943538817554) q[7];
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
ry(-0.8380248503702941) q[0];
rz(-1.3146762500503775) q[0];
ry(-0.07969150325478934) q[1];
rz(0.2507838253802214) q[1];
ry(2.0997411319702506) q[2];
rz(-2.6007947411842265) q[2];
ry(0.22400308583844042) q[3];
rz(-0.12875922124431724) q[3];
ry(-0.03421672465248892) q[4];
rz(-3.0060349413586493) q[4];
ry(0.217892830271766) q[5];
rz(0.4988105658857622) q[5];
ry(-2.803550917586392) q[6];
rz(-2.518451277520512) q[6];
ry(-1.7522422202337777) q[7];
rz(-1.1277233044315171) q[7];
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
ry(2.0787736429228327) q[0];
rz(1.471673800539085) q[0];
ry(2.8272463468986238) q[1];
rz(-1.9861180195768509) q[1];
ry(-0.8804006545339746) q[2];
rz(-1.808774731715551) q[2];
ry(-2.6618231194487034) q[3];
rz(-0.3441923305796355) q[3];
ry(-1.8110173902003872) q[4];
rz(-1.1566696678412147) q[4];
ry(-2.5087085314935322) q[5];
rz(2.401180478600778) q[5];
ry(1.206686664858296) q[6];
rz(2.3950882974967307) q[6];
ry(2.198129062468823) q[7];
rz(-1.1978301758814436) q[7];
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
ry(1.4725728939782472) q[0];
rz(-2.1651663449200145) q[0];
ry(-0.26667666735384365) q[1];
rz(2.9290744555305714) q[1];
ry(-0.5079497739831839) q[2];
rz(-0.3797896368176393) q[2];
ry(-2.7739028778494257) q[3];
rz(-3.024376647732374) q[3];
ry(-1.8878845362337897) q[4];
rz(2.0671149694050532) q[4];
ry(-3.09321917916432) q[5];
rz(-0.21703600856183503) q[5];
ry(2.9616746675214056) q[6];
rz(-1.2998236936304686) q[6];
ry(-2.6056774821188045) q[7];
rz(-2.048839580819779) q[7];
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
ry(0.6552879714080977) q[0];
rz(2.560022836682) q[0];
ry(1.6784762071107626) q[1];
rz(1.4367801070743365) q[1];
ry(0.1433726517681304) q[2];
rz(0.4030307400329826) q[2];
ry(2.6633075408889155) q[3];
rz(-2.4151757868191983) q[3];
ry(0.9558245465219154) q[4];
rz(2.714947015455933) q[4];
ry(-0.6027970340816992) q[5];
rz(-1.1099082996901912) q[5];
ry(-0.4037935861137356) q[6];
rz(0.25143313709724424) q[6];
ry(-2.394858994042509) q[7];
rz(1.535549332933586) q[7];
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
ry(0.36298904172616275) q[0];
rz(2.0005403780656037) q[0];
ry(0.9306314009063906) q[1];
rz(1.6014776598901854) q[1];
ry(-0.589374926587185) q[2];
rz(-0.830576021090589) q[2];
ry(2.4183217736585814) q[3];
rz(2.3191726455143473) q[3];
ry(-1.975295444946294) q[4];
rz(-3.10282939725946) q[4];
ry(1.2560579932153162) q[5];
rz(1.7437499377711012) q[5];
ry(2.144701126963372) q[6];
rz(-3.049913552292953) q[6];
ry(2.3227133149033383) q[7];
rz(-2.2622676474213095) q[7];
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
ry(-1.0289437453791457) q[0];
rz(-0.5105252124399232) q[0];
ry(0.7079395551956437) q[1];
rz(-2.445549998829719) q[1];
ry(-2.3710217780174934) q[2];
rz(-0.33816689953139534) q[2];
ry(2.090038002092718) q[3];
rz(0.0062746312108690186) q[3];
ry(-0.9834910994559207) q[4];
rz(-1.9038129731973865) q[4];
ry(2.5289281506779133) q[5];
rz(0.7451832201741072) q[5];
ry(-0.8566693014666364) q[6];
rz(0.24357547590756923) q[6];
ry(0.35498935801617737) q[7];
rz(1.5707168240977598) q[7];
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
ry(1.8627198349305916) q[0];
rz(1.84512818185263) q[0];
ry(0.1537009545339938) q[1];
rz(-0.9715707307991064) q[1];
ry(1.8225041804289654) q[2];
rz(2.1645106519158066) q[2];
ry(-2.242457774612068) q[3];
rz(1.3327355608214284) q[3];
ry(-2.0470551031485384) q[4];
rz(2.1399443981294537) q[4];
ry(0.6276311078768222) q[5];
rz(-2.4956580157040116) q[5];
ry(-0.30482548890616334) q[6];
rz(-3.043906117269026) q[6];
ry(0.22311261655853395) q[7];
rz(-2.6230668230999474) q[7];