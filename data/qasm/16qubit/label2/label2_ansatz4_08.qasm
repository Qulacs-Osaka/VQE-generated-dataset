OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.574002576069815) q[0];
rz(-2.2446396593363156) q[0];
ry(-1.5725418926325367) q[1];
rz(0.004462144852506112) q[1];
ry(-1.3501733330010255) q[2];
rz(-1.4367663842175802) q[2];
ry(1.5533923331218054) q[3];
rz(-1.5863230929252319) q[3];
ry(-0.0002165111622507276) q[4];
rz(-1.751642518572769) q[4];
ry(3.1415802087403137) q[5];
rz(2.1553793092987648) q[5];
ry(-1.5948244693887494) q[6];
rz(-2.6102915798628192) q[6];
ry(-1.5770495656904007) q[7];
rz(1.8516384934593884) q[7];
ry(1.5690631862337407) q[8];
rz(2.1632688611580133) q[8];
ry(-1.566702203992162) q[9];
rz(0.09419421586643378) q[9];
ry(-1.57064777638558) q[10];
rz(0.7799475377170991) q[10];
ry(1.5703358224659738) q[11];
rz(0.15522667570896242) q[11];
ry(2.624659247668616) q[12];
rz(-2.598862593727584) q[12];
ry(1.569445100819513) q[13];
rz(-0.00901246352365259) q[13];
ry(-1.5633338751101782) q[14];
rz(-0.019872979163744888) q[14];
ry(1.5625199070839297) q[15];
rz(-2.8610462867403657) q[15];
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
ry(-0.04128072708455299) q[0];
rz(0.5311966028995085) q[0];
ry(-2.312589828697835) q[1];
rz(-1.5330439001427694) q[1];
ry(-1.4917111446431874) q[2];
rz(0.13410589620602129) q[2];
ry(-1.5836921617753266) q[3];
rz(-0.6749233781972492) q[3];
ry(-3.1112792016292254) q[4];
rz(1.8682844644787346) q[4];
ry(0.02957315511422642) q[5];
rz(0.4350299296457232) q[5];
ry(-3.1329264279671567) q[6];
rz(-2.449470695975738) q[6];
ry(-0.04546485169766257) q[7];
rz(-1.6617151757121302) q[7];
ry(1.5173809376809488) q[8];
rz(-3.0601263133595897) q[8];
ry(-1.472986058786135) q[9];
rz(-0.009352843241873465) q[9];
ry(-0.06565758977857417) q[10];
rz(-2.3621828414067285) q[10];
ry(-0.011330705593301382) q[11];
rz(1.4664355416967987) q[11];
ry(1.570066047560439) q[12];
rz(1.571720775087277) q[12];
ry(1.5690447254806346) q[13];
rz(0.3901021079787035) q[13];
ry(-1.0107766578405344) q[14];
rz(1.3946937248039752) q[14];
ry(-0.5674842639472808) q[15];
rz(-1.8558847891311234) q[15];
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
ry(-0.07849810961820981) q[0];
rz(-0.39095553499575664) q[0];
ry(2.317972820871839) q[1];
rz(-0.5574262620729895) q[1];
ry(-1.536126104599405) q[2];
rz(0.10511916149531064) q[2];
ry(1.6129522795151563) q[3];
rz(-2.83299987585796) q[3];
ry(-3.1414676436050875) q[4];
rz(0.6177124077413678) q[4];
ry(-0.0006996478285001828) q[5];
rz(-3.131156770412005) q[5];
ry(3.0857594797447208) q[6];
rz(-2.1439102224554567) q[6];
ry(-3.1353456107282147) q[7];
rz(1.6957340297665384) q[7];
ry(-1.5723965483899323) q[8];
rz(-2.6419857015354116) q[8];
ry(-1.5692912545981799) q[9];
rz(-3.114755862785817) q[9];
ry(3.135393609566936) q[10];
rz(-1.7298809628284877) q[10];
ry(-3.135333818121966) q[11];
rz(0.8991984116996559) q[11];
ry(3.1408554728730675) q[12];
rz(-3.140079722841574) q[12];
ry(-3.138964234147415) q[13];
rz(-1.1809789485718871) q[13];
ry(0.0040581820363248386) q[14];
rz(-1.3803098964470568) q[14];
ry(1.6188859788918415) q[15];
rz(0.8124723190505) q[15];
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
ry(-1.0255495290128929) q[0];
rz(1.9069370296898098) q[0];
ry(-2.018847667751639) q[1];
rz(-1.3599283837741885) q[1];
ry(-1.9042209631027909) q[2];
rz(-1.9044451235394728) q[2];
ry(2.74661282783134) q[3];
rz(1.826876494490278) q[3];
ry(0.03512111232027859) q[4];
rz(-0.5178324101851123) q[4];
ry(-3.0910612646604165) q[5];
rz(-1.5260618512592945) q[5];
ry(2.924979817729812) q[6];
rz(-0.08226042876478967) q[6];
ry(2.6432251694346687) q[7];
rz(-2.241888184026255) q[7];
ry(-0.19353540572499384) q[8];
rz(0.42087274847490275) q[8];
ry(-1.5847593218391862) q[9];
rz(-2.320236970576849) q[9];
ry(3.1414661620931055) q[10];
rz(-2.2552734955065814) q[10];
ry(3.141252290414778) q[11];
rz(0.6667336782495038) q[11];
ry(-1.5810697126964846) q[12];
rz(0.048859201846386036) q[12];
ry(-1.6143582109626093) q[13];
rz(-0.0014251738565418221) q[13];
ry(2.0070481501282194) q[14];
rz(-2.3746498401411977) q[14];
ry(-0.20827523568351342) q[15];
rz(2.0371712460800016) q[15];
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
ry(-0.30564880988653276) q[0];
rz(-0.6688741804182339) q[0];
ry(1.554377901744071) q[1];
rz(1.0524598067701771) q[1];
ry(-2.731292533028999) q[2];
rz(1.8520516617298899) q[2];
ry(0.4599183139576485) q[3];
rz(-0.526034698119922) q[3];
ry(-0.0031703807608484467) q[4];
rz(2.4811459337864448) q[4];
ry(-3.074759834940098) q[5];
rz(1.4491153183575598) q[5];
ry(-0.013190403870541278) q[6];
rz(-0.6092594703474181) q[6];
ry(3.1409792503287215) q[7];
rz(-1.4801585925998948) q[7];
ry(3.140111676652076) q[8];
rz(-0.28723615320990126) q[8];
ry(-3.1415772227294267) q[9];
rz(-2.9617191470171518) q[9];
ry(-3.14144743396648) q[10];
rz(-2.060685005132466) q[10];
ry(3.138679305901938) q[11];
rz(1.3906104400781094) q[11];
ry(0.05966932785032508) q[12];
rz(3.1004673635586526) q[12];
ry(1.580776288351481) q[13];
rz(-3.138241275028214) q[13];
ry(0.44871207494262233) q[14];
rz(2.040743947062314) q[14];
ry(-1.2711871081724535) q[15];
rz(1.990773732451603) q[15];
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
ry(-3.1283292661420123) q[0];
rz(3.110168887658294) q[0];
ry(-2.933141157008653) q[1];
rz(1.2298407948987702) q[1];
ry(-3.1408667808411344) q[2];
rz(0.6270500913196058) q[2];
ry(-0.003612624588565229) q[3];
rz(2.323142380269828) q[3];
ry(1.5835276883095064) q[4];
rz(-0.3920584192087553) q[4];
ry(1.598382333500715) q[5];
rz(2.422035230313912) q[5];
ry(2.3249791020212074) q[6];
rz(0.31433219560860165) q[6];
ry(1.0653777082318419) q[7];
rz(1.6384642459398346) q[7];
ry(1.811900881163966) q[8];
rz(1.4781073618531682) q[8];
ry(0.40283644487139925) q[9];
rz(2.968433376741753) q[9];
ry(1.8737736952311357) q[10];
rz(-1.5723257824769612) q[10];
ry(1.9038455526447633) q[11];
rz(-1.5693670375172553) q[11];
ry(-1.521701081172587) q[12];
rz(1.573725800218626) q[12];
ry(-2.7322168231377657) q[13];
rz(1.5756203984026058) q[13];
ry(-2.8088019556009165) q[14];
rz(2.0454754741714636) q[14];
ry(0.33349319586624154) q[15];
rz(-0.09222254160949853) q[15];
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
ry(-0.21490800332327434) q[0];
rz(1.9036192099657459) q[0];
ry(-1.6490247307917114) q[1];
rz(2.7290619090237427) q[1];
ry(0.7903001553964204) q[2];
rz(0.6955690505635452) q[2];
ry(-2.350912690729124) q[3];
rz(1.200677979072923) q[3];
ry(-0.002338480163455072) q[4];
rz(-1.1486383441554116) q[4];
ry(3.1015243951395637) q[5];
rz(-2.3161593797713245) q[5];
ry(1.5862863470776232) q[6];
rz(-0.014531719729896862) q[6];
ry(-1.5553058036411231) q[7];
rz(-3.1271710098851924) q[7];
ry(0.06691152164228065) q[8];
rz(-1.3354787476665828) q[8];
ry(-3.141568298193317) q[9];
rz(-0.22948051530393976) q[9];
ry(0.42640230405110735) q[10];
rz(2.3764670479854435) q[10];
ry(-2.7133939857838305) q[11];
rz(-0.04290611329692152) q[11];
ry(1.57975731304573) q[12];
rz(2.210305014549932) q[12];
ry(1.571569503735554) q[13];
rz(1.6674613713157216) q[13];
ry(-2.8406068389655026) q[14];
rz(-2.643253310893303) q[14];
ry(-3.0348910724184806) q[15];
rz(-3.0491337201746975) q[15];
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
ry(-3.0712273425707366) q[0];
rz(-1.931037902942168) q[0];
ry(-1.6397071067835824) q[1];
rz(2.148021652731418) q[1];
ry(3.1084361085709062) q[2];
rz(1.8576202159451132) q[2];
ry(-0.047523111899073114) q[3];
rz(-1.3288903775091327) q[3];
ry(-1.5027718062336215) q[4];
rz(-1.5065853614505036) q[4];
ry(1.5329988603750317) q[5];
rz(1.5011526454153676) q[5];
ry(1.559128289133516) q[6];
rz(-2.6856957523715215) q[6];
ry(-1.5816404893792617) q[7];
rz(-2.66351302045272) q[7];
ry(-3.1159596949206563) q[8];
rz(-1.3366506317592792) q[8];
ry(1.5579044841094962) q[9];
rz(-0.006585128466708742) q[9];
ry(1.541930337117317) q[10];
rz(-0.7101333285707527) q[10];
ry(0.7621899595933472) q[11];
rz(2.894312348491403) q[11];
ry(0.002156957228408807) q[12];
rz(0.23832617139503381) q[12];
ry(-2.981308976080157) q[13];
rz(2.648213485596755) q[13];
ry(-0.33415072800987566) q[14];
rz(-2.9432951922998094) q[14];
ry(2.5287799673933) q[15];
rz(1.5627716240712148) q[15];
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
ry(-0.008792383583361152) q[0];
rz(1.9695077301124846) q[0];
ry(-0.015234883324855453) q[1];
rz(-0.6748909995831881) q[1];
ry(0.001340307264663047) q[2];
rz(0.2968848245251703) q[2];
ry(-0.04824567754859432) q[3];
rz(1.8085326028576143) q[3];
ry(2.307059137894243) q[4];
rz(0.2831723973548889) q[4];
ry(0.774408361859833) q[5];
rz(-1.6155058232347281) q[5];
ry(-0.6416780858774389) q[6];
rz(1.346652807671882) q[6];
ry(-2.290187915147252) q[7];
rz(1.233973333106202) q[7];
ry(-1.5778825051234817) q[8];
rz(0.1487200183311375) q[8];
ry(1.561162240413931) q[9];
rz(0.7978146831334456) q[9];
ry(0.005136460487540424) q[10];
rz(-0.4196134520695102) q[10];
ry(-0.004770691254655131) q[11];
rz(0.5258333835893598) q[11];
ry(3.1201963257880374) q[12];
rz(-0.9847612750078341) q[12];
ry(-3.130429543476031) q[13];
rz(-2.4421106981395684) q[13];
ry(2.096178824386886) q[14];
rz(0.8639775372514936) q[14];
ry(2.2158450725087295) q[15];
rz(-1.2527673462559967) q[15];
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
ry(1.6588525075128686) q[0];
rz(2.7025790629002246) q[0];
ry(0.08616213509079085) q[1];
rz(-1.5979752766115025) q[1];
ry(1.645452581277464) q[2];
rz(2.19252574829817) q[2];
ry(-1.6441492850641597) q[3];
rz(1.9975240856930654) q[3];
ry(-3.1384396504392114) q[4];
rz(0.8838576271730073) q[4];
ry(0.04712803846855707) q[5];
rz(1.8861212432348164) q[5];
ry(0.781595486371151) q[6];
rz(0.14114863123830965) q[6];
ry(-2.3318603051131803) q[7];
rz(-0.2523319672304672) q[7];
ry(-0.05592812011832263) q[8];
rz(2.769874231210298) q[8];
ry(3.131075462173719) q[9];
rz(0.3393765686990638) q[9];
ry(0.0028877290915758635) q[10];
rz(1.0056537586034322) q[10];
ry(-3.136579359704972) q[11];
rz(-2.8966032845352427) q[11];
ry(-3.1401075663378673) q[12];
rz(-0.4410052227048846) q[12];
ry(3.0280434917459536) q[13];
rz(2.072712448413834) q[13];
ry(1.5190014901518438) q[14];
rz(-2.917566961403369) q[14];
ry(-0.7060049369436134) q[15];
rz(2.806486085840406) q[15];
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
ry(3.1285889526938533) q[0];
rz(-1.9091601644415324) q[0];
ry(-1.5700668315155113) q[1];
rz(-2.685358498603161) q[1];
ry(3.1393747879647504) q[2];
rz(2.2630563780450736) q[2];
ry(-3.1402477623158163) q[3];
rz(1.929659846234927) q[3];
ry(0.023666935375515227) q[4];
rz(2.016337311071371) q[4];
ry(-0.024288220260747728) q[5];
rz(-3.0066209374876567) q[5];
ry(1.5633790551494604) q[6];
rz(-2.3299051961393604) q[6];
ry(1.6011922347369223) q[7];
rz(2.331946978120733) q[7];
ry(0.004720158267597582) q[8];
rz(0.21702053949572886) q[8];
ry(0.016385191707357194) q[9];
rz(2.024610839330527) q[9];
ry(-0.002275418932218434) q[10];
rz(3.0689037474354635) q[10];
ry(0.0011348465891876103) q[11];
rz(-1.5328920507501755) q[11];
ry(0.006706928740320528) q[12];
rz(-1.4293329281234428) q[12];
ry(3.1412081246064143) q[13];
rz(-1.8219246539028673) q[13];
ry(1.4610607594892773) q[14];
rz(-0.01525791572401839) q[14];
ry(1.5348739404771967) q[15];
rz(1.1407780159820158) q[15];
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
ry(1.6057160194831295) q[0];
rz(0.42106261845953846) q[0];
ry(3.09287332983663) q[1];
rz(2.441378784615269) q[1];
ry(-0.675949162954918) q[2];
rz(-1.2027749960667815) q[2];
ry(-2.4625041329553814) q[3];
rz(1.9463538724665863) q[3];
ry(-3.1377215654148465) q[4];
rz(-0.07261720672428454) q[4];
ry(3.1413460102923683) q[5];
rz(-2.373867640705909) q[5];
ry(1.370233425789329) q[6];
rz(0.544541471073047) q[6];
ry(-1.6806473971414764) q[7];
rz(-2.65886143426276) q[7];
ry(1.5772243803276964) q[8];
rz(1.9565727072536545) q[8];
ry(1.5631883045610324) q[9];
rz(1.9582271902675525) q[9];
ry(0.010868774583948415) q[10];
rz(2.1348394890846474) q[10];
ry(-0.01341612445824758) q[11];
rz(-1.2364184285481006) q[11];
ry(-1.5901621327761628) q[12];
rz(-2.7727269990216397) q[12];
ry(-0.22321970248305067) q[13];
rz(-1.7177605852348514) q[13];
ry(2.8632365692442776) q[14];
rz(-2.4584711460998285) q[14];
ry(-0.3487210195975931) q[15];
rz(-0.47342289532930054) q[15];