OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.0718019836078847) q[0];
rz(0.1828300174101607) q[0];
ry(-2.5927079468196053) q[1];
rz(-1.0783305911369885) q[1];
ry(-2.4284077837589417) q[2];
rz(-0.2389514307912091) q[2];
ry(0.896430658642972) q[3];
rz(2.061605619037633) q[3];
ry(0.010180985343807503) q[4];
rz(0.9089458630539096) q[4];
ry(1.3924215144755037) q[5];
rz(-2.790261781392398) q[5];
ry(0.4532045326466365) q[6];
rz(1.7595329252721723) q[6];
ry(-2.0532687589054657) q[7];
rz(1.9334082806558621) q[7];
ry(-2.507478803914345) q[8];
rz(0.45014022364825723) q[8];
ry(-1.2149562690081923) q[9];
rz(-2.562432324856308) q[9];
ry(1.6889997667722816) q[10];
rz(-0.9694083941104116) q[10];
ry(-2.9925548379725773) q[11];
rz(1.3240351372192207) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.2416785186003403) q[0];
rz(-1.885119814356604) q[0];
ry(-0.11403139222363468) q[1];
rz(-0.620884172812837) q[1];
ry(-0.4456314758117736) q[2];
rz(-1.25412267223843) q[2];
ry(-2.1266906573093367) q[3];
rz(1.175750340291261) q[3];
ry(0.03996824610514871) q[4];
rz(-1.707724582943083) q[4];
ry(-2.046598858937544) q[5];
rz(-1.6808841498681353) q[5];
ry(-0.017722466229286304) q[6];
rz(1.5523798528925765) q[6];
ry(-3.13394801025821) q[7];
rz(1.9647823634044403) q[7];
ry(2.0639049878832507) q[8];
rz(1.8058047523608824) q[8];
ry(1.3928358424350264) q[9];
rz(-1.8331383426920445) q[9];
ry(-0.7958806521527091) q[10];
rz(2.5695040294291855) q[10];
ry(0.5567798559636029) q[11];
rz(3.1052922599200885) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.383160171957809) q[0];
rz(-2.255027534713619) q[0];
ry(3.097138322688796) q[1];
rz(-2.3891835176624747) q[1];
ry(0.2676085688527605) q[2];
rz(1.524099622530561) q[2];
ry(0.7690915940712462) q[3];
rz(-1.5642832935366038) q[3];
ry(-0.008314471574456308) q[4];
rz(0.3880879735446901) q[4];
ry(-0.012995682255752209) q[5];
rz(0.3539886312048495) q[5];
ry(-1.9819776288635271) q[6];
rz(0.46003059363051074) q[6];
ry(1.2310598803438653) q[7];
rz(-2.1310391097696955) q[7];
ry(-1.5467520863570359) q[8];
rz(2.02765012454947) q[8];
ry(1.3321253370249087) q[9];
rz(-2.6836926834985757) q[9];
ry(-0.5150233500221669) q[10];
rz(-2.944646695614713) q[10];
ry(1.1450538848380898) q[11];
rz(-0.972324498295056) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.415032245312202) q[0];
rz(2.487718620143357) q[0];
ry(-0.331815375782948) q[1];
rz(-0.8977179630548422) q[1];
ry(1.6088589938111169) q[2];
rz(-1.8101044613426582) q[2];
ry(-1.8108599958714917) q[3];
rz(1.7775921102407988) q[3];
ry(-1.9714100087438462) q[4];
rz(1.3746621325144652) q[4];
ry(-2.7614765547971847) q[5];
rz(-0.753720614665788) q[5];
ry(-2.3773518900756834) q[6];
rz(2.4854885156959834) q[6];
ry(2.5423086826782666) q[7];
rz(-0.9908945648943949) q[7];
ry(-2.482826854870663) q[8];
rz(0.24601409238308228) q[8];
ry(-2.217861631289161) q[9];
rz(-0.07205937123424722) q[9];
ry(-0.010201658522274393) q[10];
rz(-3.132807138495144) q[10];
ry(-2.5366730658709082) q[11];
rz(-0.9150868141566589) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.052216315830362) q[0];
rz(-0.9505744698206922) q[0];
ry(1.2904037386033673) q[1];
rz(0.2578538522450008) q[1];
ry(2.1721963752919775) q[2];
rz(2.5804476198466815) q[2];
ry(-3.0907356105164845) q[3];
rz(2.1102577539392167) q[3];
ry(-3.1229838189021395) q[4];
rz(2.7729717176689954) q[4];
ry(-1.8067559519620264) q[5];
rz(-0.0008089282125363083) q[5];
ry(2.8725747655026743) q[6];
rz(2.592124959836251) q[6];
ry(0.03309642585045136) q[7];
rz(0.2904444562711834) q[7];
ry(0.7467838574400124) q[8];
rz(-1.7990529647906683) q[8];
ry(2.534766793242763) q[9];
rz(2.4644474536416374) q[9];
ry(0.03651933317235687) q[10];
rz(-2.1644807596829345) q[10];
ry(-1.1582131266712743) q[11];
rz(-0.2681630142048348) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.7982980084017837) q[0];
rz(2.0913624944567806) q[0];
ry(0.4053092024145453) q[1];
rz(-0.12852695683417004) q[1];
ry(-1.8314996399645578) q[2];
rz(-3.021000226711252) q[2];
ry(-1.0354653469020958) q[3];
rz(0.6932841231272115) q[3];
ry(1.4028576202606207) q[4];
rz(-0.1299395035355051) q[4];
ry(-0.5918846876024044) q[5];
rz(2.6190549646791252) q[5];
ry(3.1363090477759434) q[6];
rz(-2.529652107931226) q[6];
ry(-1.9993566740660436) q[7];
rz(2.4167866451912383) q[7];
ry(-1.2448210557852892) q[8];
rz(-2.3954266153416572) q[8];
ry(1.8218958405774235) q[9];
rz(-2.9456338402928046) q[9];
ry(-0.5072448954945586) q[10];
rz(-1.789686021783674) q[10];
ry(0.33015174226628563) q[11];
rz(0.10625220677503311) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.2557278517150703) q[0];
rz(0.8598759791812594) q[0];
ry(-1.3267633625314779) q[1];
rz(1.2764629163225476) q[1];
ry(0.8511020280424662) q[2];
rz(2.6652376391416963) q[2];
ry(3.1118376842860367) q[3];
rz(-0.7593577422288769) q[3];
ry(-0.048827875763876705) q[4];
rz(-1.9733105794086228) q[4];
ry(-2.050796833100241) q[5];
rz(1.984669674490763) q[5];
ry(-1.2348957195965755) q[6];
rz(-0.3330213789583431) q[6];
ry(3.0416129618142316) q[7];
rz(0.4564942056575089) q[7];
ry(-0.030483844050715518) q[8];
rz(2.846647488582468) q[8];
ry(-1.0878910663170862) q[9];
rz(-0.4908037849998585) q[9];
ry(-1.5920337343467503) q[10];
rz(-0.002223509332115014) q[10];
ry(0.02333170574302108) q[11];
rz(1.8844243781237038) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.0480179778327514) q[0];
rz(-1.9603808499639737) q[0];
ry(-2.6834862396248513) q[1];
rz(-2.9442225002918816) q[1];
ry(1.02524382248306) q[2];
rz(2.726721690204436) q[2];
ry(-0.6343097830548885) q[3];
rz(1.0595749653866742) q[3];
ry(-0.568552801703933) q[4];
rz(1.6849058504871381) q[4];
ry(-0.008700357855283152) q[5];
rz(2.4549693221905655) q[5];
ry(-3.14127858404589) q[6];
rz(-2.0713822206632795) q[6];
ry(2.0515914051750697) q[7];
rz(-0.493524304216117) q[7];
ry(-0.7260084060850596) q[8];
rz(-3.079597971032784) q[8];
ry(1.1599763144927018) q[9];
rz(-1.809334626941772) q[9];
ry(1.659407290348732) q[10];
rz(2.4654447200061695) q[10];
ry(-1.49226366115614) q[11];
rz(2.4454446334167965) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.670161664359991) q[0];
rz(0.2664059293135132) q[0];
ry(2.3556811684475125) q[1];
rz(0.9810716426350544) q[1];
ry(0.5765866981011605) q[2];
rz(-1.496239427401033) q[2];
ry(-2.279268519687534) q[3];
rz(3.0893829977213705) q[3];
ry(3.109117971834541) q[4];
rz(2.97651738560845) q[4];
ry(1.2145992055715782) q[5];
rz(-1.3209670695169204) q[5];
ry(2.110816577201809) q[6];
rz(-1.6708269310269936) q[6];
ry(2.1407254066060566) q[7];
rz(0.045673929060929375) q[7];
ry(-2.220238792093486) q[8];
rz(-0.1852408666334) q[8];
ry(1.53426411692618) q[9];
rz(-1.6288497419426546) q[9];
ry(2.306622523352411) q[10];
rz(1.455679887464344) q[10];
ry(-1.2264035747247777) q[11];
rz(-1.2473438077525643) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.5050296810016429) q[0];
rz(-1.091231691840945) q[0];
ry(-0.039377696248437566) q[1];
rz(-1.5955934582192521) q[1];
ry(-2.1403384309331663) q[2];
rz(-2.46618231428285) q[2];
ry(2.098057599997826) q[3];
rz(3.104284192028161) q[3];
ry(-3.1291577980452536) q[4];
rz(0.37905603637009433) q[4];
ry(0.002791501714615495) q[5];
rz(0.9711156186476902) q[5];
ry(2.328003198003757) q[6];
rz(3.1122727525035168) q[6];
ry(2.226729871875768) q[7];
rz(-0.6526564657875662) q[7];
ry(-1.8665917946230204) q[8];
rz(1.4614638189353153) q[8];
ry(-2.423668515157487) q[9];
rz(-0.7368571616128184) q[9];
ry(1.2221351860894947) q[10];
rz(-2.936571393128419) q[10];
ry(-0.3047094962140278) q[11];
rz(-1.0278408371866181) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.02974819532186057) q[0];
rz(-1.1731125638318165) q[0];
ry(1.5899743431324749) q[1];
rz(-1.7296987136718034) q[1];
ry(-0.030952383737158584) q[2];
rz(2.59785330449534) q[2];
ry(2.060788641195171) q[3];
rz(-0.47232257070037365) q[3];
ry(2.416885306307852) q[4];
rz(-3.0384274444420143) q[4];
ry(-0.00130269397840177) q[5];
rz(1.9929644488062106) q[5];
ry(-2.2487529978253527) q[6];
rz(2.573741741695307) q[6];
ry(-0.028093065131174433) q[7];
rz(0.47107424788738056) q[7];
ry(0.10147580403724543) q[8];
rz(-1.506917224539161) q[8];
ry(-0.04916237230851994) q[9];
rz(2.1075418239975674) q[9];
ry(-1.1880904960282468) q[10];
rz(-2.6157167021714267) q[10];
ry(-1.2476353540437484) q[11];
rz(2.609849270774708) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9151952992386065) q[0];
rz(1.8834481430877172) q[0];
ry(2.0319453173422373) q[1];
rz(0.0018363014534235936) q[1];
ry(-1.2778163969820422) q[2];
rz(1.492740570423969) q[2];
ry(2.911702737187905) q[3];
rz(-2.876732844965514) q[3];
ry(0.11116465348314547) q[4];
rz(3.117425768624563) q[4];
ry(-1.058870124294877) q[5];
rz(1.280767110543798) q[5];
ry(0.8058759724482698) q[6];
rz(0.3265162128555866) q[6];
ry(1.5713662679586733) q[7];
rz(3.049393648500514) q[7];
ry(1.4865683541744967) q[8];
rz(-2.3113779292774765) q[8];
ry(-0.03935021302335947) q[9];
rz(-1.6323725055712013) q[9];
ry(-1.333261843851644) q[10];
rz(-1.2218630287682313) q[10];
ry(2.964277920195779) q[11];
rz(2.6619011502596894) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.026171626431590767) q[0];
rz(-0.6119646679424786) q[0];
ry(-3.096220603239987) q[1];
rz(1.5663234860347606) q[1];
ry(3.13328979466478) q[2];
rz(-1.717830113440219) q[2];
ry(-0.0015833217737668546) q[3];
rz(1.2129365873527638) q[3];
ry(0.16536448240795848) q[4];
rz(1.305135185196809) q[4];
ry(-0.0007710707054089383) q[5];
rz(1.9028723079700152) q[5];
ry(3.1366389986416774) q[6];
rz(-1.435706960933543) q[6];
ry(0.026760579098482243) q[7];
rz(-2.9913667856598183) q[7];
ry(0.1860756130721983) q[8];
rz(1.4009478322944133) q[8];
ry(-0.08707010130128356) q[9];
rz(-2.108648445618183) q[9];
ry(1.87916872192769) q[10];
rz(-0.8167274410260594) q[10];
ry(0.8970156253881234) q[11];
rz(-3.127066407097483) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.8735723739353132) q[0];
rz(-0.9406038857412025) q[0];
ry(-1.3322894904705198) q[1];
rz(2.810827890772907) q[1];
ry(-2.078778811285267) q[2];
rz(-1.4308353462544254) q[2];
ry(0.9537106365978322) q[3];
rz(-0.4670208402668965) q[3];
ry(-0.001950793079904289) q[4];
rz(-1.1806699192115495) q[4];
ry(-2.1961655805799043) q[5];
rz(-2.7309873795358013) q[5];
ry(-2.750981375359058) q[6];
rz(0.593875306852147) q[6];
ry(2.6321283555799075) q[7];
rz(-3.002618024222628) q[7];
ry(-0.45682098927751497) q[8];
rz(-1.551692956042649) q[8];
ry(2.3141501603070602) q[9];
rz(2.766122484595609) q[9];
ry(0.7956533352819263) q[10];
rz(1.5920898549985525) q[10];
ry(-1.7379615863249065) q[11];
rz(2.1128679014858656) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.250813981025738) q[0];
rz(-0.2249300709109825) q[0];
ry(2.84794072216013) q[1];
rz(-1.329340544082386) q[1];
ry(3.1382183106487305) q[2];
rz(-1.0603705904237066) q[2];
ry(1.5989492168453925) q[3];
rz(3.1162462559255033) q[3];
ry(-0.5371605777491629) q[4];
rz(0.09891416216695009) q[4];
ry(-3.1106339208565044) q[5];
rz(0.22946394332376507) q[5];
ry(-0.05669381193201417) q[6];
rz(2.2405056608571203) q[6];
ry(-1.9668386390573458) q[7];
rz(-0.011060707691536997) q[7];
ry(-0.3118176931726123) q[8];
rz(0.25863787357299556) q[8];
ry(3.079941028094618) q[9];
rz(-0.29064920813861495) q[9];
ry(2.6127806302628493) q[10];
rz(3.09699787933417) q[10];
ry(-1.5033215788627032) q[11];
rz(-1.8172099386660847) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.7770153220388885) q[0];
rz(1.5000205421017672) q[0];
ry(-0.9407370481332604) q[1];
rz(3.0727266956276984) q[1];
ry(-1.4423998924443346) q[2];
rz(2.8560274336907914) q[2];
ry(-0.48019819642841816) q[3];
rz(2.159993231256324) q[3];
ry(0.10486666067215278) q[4];
rz(-0.41969247737103554) q[4];
ry(2.998732823655463) q[5];
rz(2.865977418884972) q[5];
ry(3.1403702768767308) q[6];
rz(-0.3968815342740157) q[6];
ry(-1.6086155882646203) q[7];
rz(-3.0341495919886103) q[7];
ry(-0.6724319385951324) q[8];
rz(2.466663784099898) q[8];
ry(-1.8733816060781434) q[9];
rz(0.15732284259762663) q[9];
ry(-2.4752545716681835) q[10];
rz(-0.31334370236239495) q[10];
ry(2.6057096202615133) q[11];
rz(1.3615811768555293) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.034656111834192316) q[0];
rz(3.0465427006311723) q[0];
ry(1.556090683060824) q[1];
rz(0.9246165289136175) q[1];
ry(3.1363186116209936) q[2];
rz(0.27232026516015745) q[2];
ry(0.7501533898595695) q[3];
rz(3.0248042577992607) q[3];
ry(1.1023379308819978) q[4];
rz(1.5278142723622976) q[4];
ry(1.6501806235007095) q[5];
rz(-0.6060042265801533) q[5];
ry(0.01047696468936099) q[6];
rz(0.402232883076068) q[6];
ry(-2.06811813093475) q[7];
rz(-2.775707360693635) q[7];
ry(-2.567304557111073) q[8];
rz(-1.6211424626775761) q[8];
ry(-0.04982293967278206) q[9];
rz(2.820701944083318) q[9];
ry(2.7802266816380135) q[10];
rz(-2.19815530925825) q[10];
ry(0.9338550294106889) q[11];
rz(2.2135384511659826) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.445110668984844) q[0];
rz(2.5618347061154148) q[0];
ry(-0.635730645111118) q[1];
rz(2.5228851369807574) q[1];
ry(-1.7729168513678992) q[2];
rz(-3.094199510575885) q[2];
ry(-2.899973644297988) q[3];
rz(3.0645017657581253) q[3];
ry(0.01537970104761648) q[4];
rz(2.0531505607326253) q[4];
ry(-3.1271616710469807) q[5];
rz(1.0523117293405546) q[5];
ry(-2.7338481598560924) q[6];
rz(-1.2490993728467803) q[6];
ry(0.9452830993892616) q[7];
rz(0.19401835961765368) q[7];
ry(-1.649202461420852) q[8];
rz(0.7871818479389487) q[8];
ry(-1.4010352962667594) q[9];
rz(1.5579966064383601) q[9];
ry(0.9509413269827517) q[10];
rz(1.5572438688749957) q[10];
ry(2.7607477543616716) q[11];
rz(2.804093189809433) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.09773322106058213) q[0];
rz(1.1264830705818172) q[0];
ry(0.9236810500760667) q[1];
rz(-1.731077751547215) q[1];
ry(2.076871153540423) q[2];
rz(0.07489705103484312) q[2];
ry(-1.336858276251322) q[3];
rz(-1.2925553786374835) q[3];
ry(-0.37715897168535323) q[4];
rz(2.0116671393043557) q[4];
ry(-0.894374280460209) q[5];
rz(-0.12454637387831814) q[5];
ry(3.1412526947728985) q[6];
rz(1.8965182502429294) q[6];
ry(-0.4686398109858771) q[7];
rz(0.2234458373104653) q[7];
ry(-3.067298237013159) q[8];
rz(-0.07967174168819252) q[8];
ry(0.759500150849334) q[9];
rz(-0.7179916997150456) q[9];
ry(-2.3835779467982094) q[10];
rz(-2.1170573603121436) q[10];
ry(1.0090448360919502) q[11];
rz(-1.5499118505413314) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.1799228234729857) q[0];
rz(0.029561579394349554) q[0];
ry(-3.1147684266485465) q[1];
rz(1.462214847234777) q[1];
ry(-2.5184700653893706) q[2];
rz(-3.1394655661393225) q[2];
ry(-3.1374128036327145) q[3];
rz(-2.9197438176402173) q[3];
ry(-3.1410811664940366) q[4];
rz(-0.7349584475321471) q[4];
ry(1.8349154781695187) q[5];
rz(-3.098951466997937) q[5];
ry(-2.309613207809725) q[6];
rz(1.1214337226460547) q[6];
ry(2.4796180410240103) q[7];
rz(-3.107867431699516) q[7];
ry(-3.132730786364649) q[8];
rz(-2.763671086167583) q[8];
ry(-2.081522409502874) q[9];
rz(2.215292790733053) q[9];
ry(-2.483019470349645) q[10];
rz(-1.3622638566708343) q[10];
ry(0.8382420683340832) q[11];
rz(2.7330317421107) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.1352705053837915) q[0];
rz(0.9661060492605132) q[0];
ry(-1.3355478744750282) q[1];
rz(1.5637605505295564) q[1];
ry(2.3476084628714013) q[2];
rz(3.0610554463607955) q[2];
ry(1.6458048760946962) q[3];
rz(-2.8724292759440755) q[3];
ry(2.699750262071435) q[4];
rz(-0.0023953996934670536) q[4];
ry(2.028489775307894) q[5];
rz(-3.0091837036319724) q[5];
ry(3.1354709144522293) q[6];
rz(-1.0445211452896537) q[6];
ry(-3.037138865081282) q[7];
rz(-0.5792861357060817) q[7];
ry(-0.06609713974290177) q[8];
rz(-1.378700860392162) q[8];
ry(-1.4014250791558522) q[9];
rz(2.9200116761859123) q[9];
ry(-1.411182903279411) q[10];
rz(1.4720861518023713) q[10];
ry(0.6410778007548917) q[11];
rz(-3.102441015034719) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.7298340038695068) q[0];
rz(1.0261806286643558) q[0];
ry(2.9590823081948496) q[1];
rz(-2.681432687826947) q[1];
ry(2.794879675607776) q[2];
rz(-3.0329015394074927) q[2];
ry(0.0012751320915525) q[3];
rz(1.3520107383331454) q[3];
ry(1.5681802350309804) q[4];
rz(0.008066660214550903) q[4];
ry(2.8609100608687377) q[5];
rz(0.3142905579212731) q[5];
ry(-0.002782278809465907) q[6];
rz(0.7891274833400068) q[6];
ry(-1.5300470295211879) q[7];
rz(2.490577338892307) q[7];
ry(-3.1278431025810747) q[8];
rz(-1.7819019095273594) q[8];
ry(1.2517185257654309) q[9];
rz(2.214940163092336) q[9];
ry(2.617698398479435) q[10];
rz(0.26608112856262794) q[10];
ry(-0.7109892964716007) q[11];
rz(-1.1708778082538496) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.3494358083542248) q[0];
rz(0.2787843880276874) q[0];
ry(-0.2042874501954897) q[1];
rz(-0.8671991141332213) q[1];
ry(1.721036428217941) q[2];
rz(0.00698222626477017) q[2];
ry(-1.5979348018363686) q[3];
rz(0.002508233142938624) q[3];
ry(-3.0271912020205645) q[4];
rz(0.012650588007065077) q[4];
ry(-1.3366318808107014) q[5];
rz(1.4718697257640008) q[5];
ry(-0.034546268967454025) q[6];
rz(-2.867640630228166) q[6];
ry(0.03382650284267186) q[7];
rz(-1.3067651241439622) q[7];
ry(3.1348188097785066) q[8];
rz(-2.2247843111903647) q[8];
ry(1.5120899738522091) q[9];
rz(1.0054246713352404) q[9];
ry(-0.050758218927570375) q[10];
rz(2.6926095429715233) q[10];
ry(2.3383840122464377) q[11];
rz(-0.3593993564869928) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.014840464443323) q[0];
rz(2.2792274073741177) q[0];
ry(0.03682245963178943) q[1];
rz(-1.2177921946744554) q[1];
ry(-0.08911673261998221) q[2];
rz(3.1237813800372716) q[2];
ry(0.7959968079194014) q[3];
rz(-0.981973168452674) q[3];
ry(-0.3838671033668774) q[4];
rz(-0.005062321569842479) q[4];
ry(-0.00018665738495040533) q[5];
rz(1.7043600099932714) q[5];
ry(0.0009615281559023359) q[6];
rz(1.1113972941506163) q[6];
ry(0.5041706837742244) q[7];
rz(-2.4048057025232703) q[7];
ry(0.0020619459639119597) q[8];
rz(1.835234304584316) q[8];
ry(-1.0515752258026037) q[9];
rz(-1.2968791860891553) q[9];
ry(0.8843566095236357) q[10];
rz(0.4349613535193351) q[10];
ry(-1.1283630234304105) q[11];
rz(-1.6186388637222286) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.3210651779817324) q[0];
rz(2.0476702721410054) q[0];
ry(0.8868140974876626) q[1];
rz(1.97973807189909) q[1];
ry(1.5038567208175149) q[2];
rz(3.1386566671694505) q[2];
ry(3.1414255785049816) q[3];
rz(-0.9871257529891899) q[3];
ry(-0.7118846767605873) q[4];
rz(0.0004372898206863595) q[4];
ry(-2.8901746787585716) q[5];
rz(0.034123497093646364) q[5];
ry(-1.6992890213944936) q[6];
rz(-3.1091222003968744) q[6];
ry(-3.1009656973386766) q[7];
rz(-1.3391465930285635) q[7];
ry(1.5656092353036843) q[8];
rz(-1.546722177148586) q[8];
ry(0.4474352883286353) q[9];
rz(2.284611652529303) q[9];
ry(2.7320649709217513) q[10];
rz(-2.2902685412620647) q[10];
ry(2.016492173901983) q[11];
rz(1.476300320878077) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.161729375276611) q[0];
rz(-0.24922560783214465) q[0];
ry(0.018427675320107362) q[1];
rz(2.9199744339851006) q[1];
ry(1.5041257845149707) q[2];
rz(-3.122183558414883) q[2];
ry(1.5411094745705785) q[3];
rz(1.4482732056011907) q[3];
ry(1.899938220133489) q[4];
rz(3.137757194409162) q[4];
ry(2.554919912590167) q[5];
rz(-1.7251921424261614) q[5];
ry(-1.570506686169457) q[6];
rz(3.14156049737351) q[6];
ry(-0.4960592314684744) q[7];
rz(3.136262529653512) q[7];
ry(-2.315913250160187) q[8];
rz(3.1223773725847814) q[8];
ry(1.5805770620744761) q[9];
rz(-1.4040839208807956) q[9];
ry(-1.6061867807058907) q[10];
rz(1.7132262416758866) q[10];
ry(2.3637683593208343) q[11];
rz(-0.19569402413551895) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.9232778517737614) q[0];
rz(-0.10012625942357353) q[0];
ry(-0.18087648190984487) q[1];
rz(1.3456197742653728) q[1];
ry(-1.3823170659753163) q[2];
rz(-1.6880399369534356) q[2];
ry(-2.8225760763365964) q[3];
rz(-2.99163755340203) q[3];
ry(-0.7289577437449908) q[4];
rz(-2.6384828615511977) q[4];
ry(0.0016277885507046776) q[5];
rz(1.7273719252360153) q[5];
ry(-1.826088295298894) q[6];
rz(1.1468699828658249) q[6];
ry(-1.565162196010294) q[7];
rz(0.00026636613022829465) q[7];
ry(1.5656483033535311) q[8];
rz(3.1319118849314624) q[8];
ry(3.1072124847169023) q[9];
rz(1.7739844385361014) q[9];
ry(3.077391776911251) q[10];
rz(-0.8449324655106586) q[10];
ry(3.126533859435569) q[11];
rz(1.9134029643954795) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.474709993921517) q[0];
rz(-3.0811985696599122) q[0];
ry(-0.07744450705756238) q[1];
rz(1.9074017930254836) q[1];
ry(-3.1399648890742724) q[2];
rz(-1.7063864857839148) q[2];
ry(0.0008911035385184007) q[3];
rz(-1.6023837752924472) q[3];
ry(3.1385528743521958) q[4];
rz(-2.638152446953508) q[4];
ry(2.8963077077685755) q[5];
rz(0.051112906393840696) q[5];
ry(3.141538502139682) q[6];
rz(-1.9945568065820591) q[6];
ry(-0.305427653616868) q[7];
rz(-3.127258780020682) q[7];
ry(1.5655865454376754) q[8];
rz(-2.490310863243279) q[8];
ry(1.5773680254465434) q[9];
rz(3.120187475945528) q[9];
ry(-2.295898050073837) q[10];
rz(0.0740006446111275) q[10];
ry(2.570502325763951) q[11];
rz(-0.05588856239252582) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.3841853940317588) q[0];
rz(0.9562393706609906) q[0];
ry(-3.093823996706098) q[1];
rz(-1.4199306097220263) q[1];
ry(0.7322006143992752) q[2];
rz(-0.3493971353900751) q[2];
ry(-2.2179606302411554) q[3];
rz(0.39909415849214835) q[3];
ry(2.3791433857515445) q[4];
rz(0.5049355526258088) q[4];
ry(-0.07789334856937749) q[5];
rz(-0.04960374339818059) q[5];
ry(-1.607988229531787) q[6];
rz(-3.107944735535348) q[6];
ry(1.6439817857830443) q[7];
rz(-0.006737107753829535) q[7];
ry(-3.103144948223221) q[8];
rz(-2.482490910352003) q[8];
ry(-1.571096508357966) q[9];
rz(-1.045218618831667) q[9];
ry(0.19381994541885664) q[10];
rz(1.607812463983974) q[10];
ry(0.6689260103284376) q[11];
rz(-2.925450151111573) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.7729192532690976) q[0];
rz(-0.6757988020794254) q[0];
ry(1.6189152839202336) q[1];
rz(-1.5932836685178353) q[1];
ry(-3.1351123383138826) q[2];
rz(-1.957132636503813) q[2];
ry(1.5754058989141617) q[3];
rz(-1.5651457552593628) q[3];
ry(3.140501834441187) q[4];
rz(-1.0685230679496254) q[4];
ry(0.5479916650469528) q[5];
rz(1.5701395542205832) q[5];
ry(-3.1266780665282967) q[6];
rz(1.605064713354561) q[6];
ry(0.0530566063889486) q[7];
rz(1.5785607606192364) q[7];
ry(3.1161785675629656) q[8];
rz(1.5788455248043833) q[8];
ry(-0.0001764584954735741) q[9];
rz(-0.5255062930491388) q[9];
ry(1.5710529232368442) q[10];
rz(1.5707630879899304) q[10];
ry(-1.5988005641545537) q[11];
rz(3.124160960681109) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.573857618339241) q[0];
rz(2.9060166214454206) q[0];
ry(-1.5733019710909675) q[1];
rz(2.855695063988138) q[1];
ry(-1.5705877377821418) q[2];
rz(2.9519387547838205) q[2];
ry(1.5659427373873256) q[3];
rz(2.2701182413043686) q[3];
ry(1.5690617931789073) q[4];
rz(-0.19002804523003597) q[4];
ry(1.5731988789271136) q[5];
rz(2.992591993943859) q[5];
ry(-1.5711081622452987) q[6];
rz(3.136622940391993) q[6];
ry(1.5561720668701922) q[7];
rz(0.43612104561883086) q[7];
ry(1.5941058601644673) q[8];
rz(-1.714000862922979) q[8];
ry(1.570928139659041) q[9];
rz(-0.08195560053041444) q[9];
ry(1.57068334648589) q[10];
rz(1.3560229233160426) q[10];
ry(-0.0036106852529555786) q[11];
rz(1.4690709601937861) q[11];