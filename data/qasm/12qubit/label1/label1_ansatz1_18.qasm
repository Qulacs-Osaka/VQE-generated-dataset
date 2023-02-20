OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.5802519290425994) q[0];
rz(-1.4735626889329216) q[0];
ry(2.491063569266643) q[1];
rz(1.6993032876495515) q[1];
ry(-0.19685752112865718) q[2];
rz(-1.492434525081785) q[2];
ry(-0.6901069971518676) q[3];
rz(-2.7617017172836196) q[3];
ry(-0.5511144501420961) q[4];
rz(2.991394151757681) q[4];
ry(-2.1198727005842866) q[5];
rz(-1.511294374020994) q[5];
ry(0.031067271663283703) q[6];
rz(-2.117548410359883) q[6];
ry(1.3573092672645624) q[7];
rz(-0.4896307533798079) q[7];
ry(-0.09583608390054431) q[8];
rz(1.841309407025797) q[8];
ry(0.43849016776875227) q[9];
rz(-2.9360278417427534) q[9];
ry(3.1313022442902056) q[10];
rz(0.41727882563941154) q[10];
ry(2.226436678233629) q[11];
rz(0.016544172327875344) q[11];
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
ry(-1.513549019374067) q[0];
rz(-2.1267146914592043) q[0];
ry(2.9353897740004404) q[1];
rz(-1.3554020839625878) q[1];
ry(-3.1171844554377306) q[2];
rz(-0.9939072136445853) q[2];
ry(0.2730876304631807) q[3];
rz(-0.8175366467141043) q[3];
ry(0.08294079669047782) q[4];
rz(2.2420299410099283) q[4];
ry(-2.053962067374048) q[5];
rz(-1.7690218624254) q[5];
ry(1.3017735351460784) q[6];
rz(1.7952691164985035) q[6];
ry(-1.2456695299676532) q[7];
rz(1.683994262293277) q[7];
ry(-2.53787165808601) q[8];
rz(-0.6905933579412233) q[8];
ry(-0.6827629184365992) q[9];
rz(0.4312844999221479) q[9];
ry(-2.578265665661558) q[10];
rz(-0.9816116885891375) q[10];
ry(2.099461479902827) q[11];
rz(-0.839033712380032) q[11];
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
ry(1.330980963644479) q[0];
rz(2.963764776397132) q[0];
ry(1.1162805938319869) q[1];
rz(-1.0778028645372872) q[1];
ry(2.9634175593491574) q[2];
rz(0.5220330189033511) q[2];
ry(-1.8911835135888777) q[3];
rz(-1.8416886828056993) q[3];
ry(-0.47717952745869957) q[4];
rz(-1.450687807484059) q[4];
ry(-2.9822388858182465) q[5];
rz(1.6589982423832863) q[5];
ry(-0.811304993748938) q[6];
rz(2.444774636170916) q[6];
ry(-0.4372976063302003) q[7];
rz(-2.5970481298236416) q[7];
ry(-1.2959433914964862) q[8];
rz(2.6423437589465175) q[8];
ry(2.9429857004430984) q[9];
rz(0.1254950525381773) q[9];
ry(-0.4204853088136149) q[10];
rz(-1.9121414502273588) q[10];
ry(-0.1031017647161816) q[11];
rz(-0.11109929818249455) q[11];
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
ry(-0.8011433880059232) q[0];
rz(-0.057169600031669876) q[0];
ry(-2.7919179479313865) q[1];
rz(-1.891437428722294) q[1];
ry(0.03001321307450392) q[2];
rz(-1.475040585022672) q[2];
ry(2.559129498917078) q[3];
rz(-2.910891266949797) q[3];
ry(-2.7425521137187827) q[4];
rz(-2.8477222779370375) q[4];
ry(2.262810573530015) q[5];
rz(-1.124110257407807) q[5];
ry(1.4509960870104517) q[6];
rz(0.22872119490578233) q[6];
ry(-0.15333965828603893) q[7];
rz(-2.760665365942938) q[7];
ry(2.9232316195092864) q[8];
rz(1.1295753164877258) q[8];
ry(2.024496217945746) q[9];
rz(-0.7220657782645464) q[9];
ry(-0.6302544367453682) q[10];
rz(0.730493957546666) q[10];
ry(0.2548589335003309) q[11];
rz(-0.20504550186108528) q[11];
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
ry(-2.9947075422791234) q[0];
rz(-1.2534487454990118) q[0];
ry(-1.0226696844847345) q[1];
rz(-2.026797567842478) q[1];
ry(0.08382469372004309) q[2];
rz(-1.47754024982688) q[2];
ry(-2.093653477923472) q[3];
rz(-0.5663032858052333) q[3];
ry(-0.04472681253029954) q[4];
rz(1.0350449967599895) q[4];
ry(2.4916640712339726) q[5];
rz(-2.468096102052934) q[5];
ry(2.6242211925833443) q[6];
rz(2.750719143519515) q[6];
ry(-0.15823467708887673) q[7];
rz(-2.368635547970296) q[7];
ry(-0.04703851499086668) q[8];
rz(2.1891058736632507) q[8];
ry(-2.79529252713991) q[9];
rz(2.851646343577478) q[9];
ry(0.11726738880952148) q[10];
rz(2.6920704627327523) q[10];
ry(-3.0186770889778423) q[11];
rz(0.09050119176088067) q[11];
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
ry(-2.904220811648875) q[0];
rz(-1.136094584213083) q[0];
ry(-1.620952321345859) q[1];
rz(-0.36679976012918036) q[1];
ry(1.5046003357742037) q[2];
rz(1.5357700234356628) q[2];
ry(3.016091439602323) q[3];
rz(1.8029016437675975) q[3];
ry(-1.5741182196119292) q[4];
rz(0.0003565778662073526) q[4];
ry(1.731770871312218) q[5];
rz(1.5272359740965922) q[5];
ry(1.0371798375516477) q[6];
rz(-2.580517643221924) q[6];
ry(-0.0905515211998349) q[7];
rz(-1.0139076747657558) q[7];
ry(-2.695414225361042) q[8];
rz(2.3934244713910595) q[8];
ry(2.0149979537213714) q[9];
rz(2.070855886613522) q[9];
ry(2.6150358018270055) q[10];
rz(-3.0299591661293244) q[10];
ry(-2.7841972218949382) q[11];
rz(2.8985076529682163) q[11];
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
ry(-1.6161276556298079) q[0];
rz(1.487358082962139) q[0];
ry(-0.10578558394601088) q[1];
rz(-1.071906127989423) q[1];
ry(3.099384673075272) q[2];
rz(1.594144484488825) q[2];
ry(-2.931077760071244) q[3];
rz(-2.3697223747943226) q[3];
ry(-1.5552274901784708) q[4];
rz(1.8367759469013336) q[4];
ry(3.136303817040637) q[5];
rz(-2.500918372808636) q[5];
ry(1.6946393911105266) q[6];
rz(-0.09484686197508943) q[6];
ry(-0.622463635945391) q[7];
rz(3.045635800915053) q[7];
ry(-0.1520477339450066) q[8];
rz(2.2768697070406283) q[8];
ry(1.4190834182215306) q[9];
rz(-0.6549823987450618) q[9];
ry(0.010927891215132665) q[10];
rz(-2.379577759210579) q[10];
ry(-2.476232418583653) q[11];
rz(-1.5560994996567878) q[11];
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
ry(0.48144234314000206) q[0];
rz(2.5136577821593584) q[0];
ry(0.017866934141391427) q[1];
rz(2.0620686930608487) q[1];
ry(-1.7188970464583446) q[2];
rz(1.9271681461300494) q[2];
ry(-2.2407270878257677) q[3];
rz(-1.1510207697498178) q[3];
ry(-2.9840592173337415) q[4];
rz(2.8968314891388807) q[4];
ry(0.1685233011823195) q[5];
rz(-0.6840336038649325) q[5];
ry(1.920882084385885) q[6];
rz(-1.7166587177388655) q[6];
ry(-3.018869810483452) q[7];
rz(-1.7626329350508512) q[7];
ry(3.1387269818790546) q[8];
rz(-2.922819144788466) q[8];
ry(2.657726802813366) q[9];
rz(2.161728452886709) q[9];
ry(-3.04507944545355) q[10];
rz(2.2920716549879705) q[10];
ry(0.5793057425242483) q[11];
rz(-1.6482084080561596) q[11];
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
ry(1.9622405666577514) q[0];
rz(-1.5826411332952584) q[0];
ry(2.1496662873806542) q[1];
rz(-1.7559281647553644) q[1];
ry(1.752685490522723) q[2];
rz(-1.2408831914097718) q[2];
ry(1.381836962232286) q[3];
rz(2.777413231306695) q[3];
ry(-3.1227824969225493) q[4];
rz(1.243862503239786) q[4];
ry(-0.1284750750976178) q[5];
rz(-1.5792372925597151) q[5];
ry(-0.5170074853476444) q[6];
rz(-1.1533792691288651) q[6];
ry(-2.1536000832210465) q[7];
rz(1.5101544192528722) q[7];
ry(-2.620027808602583) q[8];
rz(0.40247642370314285) q[8];
ry(1.666648676170753) q[9];
rz(-1.1896748487524818) q[9];
ry(-0.1741726864669924) q[10];
rz(0.26787132209363684) q[10];
ry(0.339740184985427) q[11];
rz(0.19474990852120203) q[11];
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
ry(-2.8201524876374613) q[0];
rz(-3.0258061769473605) q[0];
ry(-2.995598284852228) q[1];
rz(-1.3152627008327615) q[1];
ry(-2.1939575733008003) q[2];
rz(0.44347138853331547) q[2];
ry(-0.33263404131690794) q[3];
rz(0.26734991005841824) q[3];
ry(3.015546148325148) q[4];
rz(-0.39417012747519764) q[4];
ry(-1.6383997642946844) q[5];
rz(2.7846742455980134) q[5];
ry(-2.1151828057505933) q[6];
rz(-3.0727397642302474) q[6];
ry(0.07987464412070722) q[7];
rz(-1.8415113211008949) q[7];
ry(-2.059754761547376) q[8];
rz(-2.0680577943386824) q[8];
ry(-2.7857584827003525) q[9];
rz(2.8382703266108233) q[9];
ry(2.9995997159615877) q[10];
rz(1.4486431192016878) q[10];
ry(2.6406928456954564) q[11];
rz(-0.7814732864092191) q[11];
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
ry(1.1640381946506624) q[0];
rz(-2.6287468175770643) q[0];
ry(3.019677824350462) q[1];
rz(2.6318280983549993) q[1];
ry(-0.3186065803895136) q[2];
rz(2.8004267901142925) q[2];
ry(2.83505499738646) q[3];
rz(2.140186534310115) q[3];
ry(-3.1182956366988797) q[4];
rz(-0.40099053021691766) q[4];
ry(3.0579491504583016) q[5];
rz(-2.008952610532157) q[5];
ry(1.6026412605380413) q[6];
rz(-2.741525285436885) q[6];
ry(-0.500332925107406) q[7];
rz(-3.0984553072038477) q[7];
ry(-0.8244913465283055) q[8];
rz(-2.901411326071889) q[8];
ry(-2.4171339036156168) q[9];
rz(-0.4348154884395141) q[9];
ry(0.6206655530915377) q[10];
rz(-0.3792382562198044) q[10];
ry(1.271086066415081) q[11];
rz(-1.3568277735677508) q[11];
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
ry(1.2576638630984975) q[0];
rz(-2.212087701428039) q[0];
ry(-0.44783599071302627) q[1];
rz(0.19588032877396588) q[1];
ry(0.22057072560208252) q[2];
rz(-2.719945623421771) q[2];
ry(-0.3150346637577961) q[3];
rz(0.3418420121167523) q[3];
ry(2.539711073736673) q[4];
rz(-1.002368136436547) q[4];
ry(-1.5341530967447294) q[5];
rz(1.0260073921818957) q[5];
ry(-0.06200444861603512) q[6];
rz(2.8301689769831024) q[6];
ry(1.7031217084354868) q[7];
rz(3.1324070560395234) q[7];
ry(0.1748100390112755) q[8];
rz(2.454507363458262) q[8];
ry(2.1749382657862553) q[9];
rz(-0.18714408053074544) q[9];
ry(-0.06060592209088302) q[10];
rz(-1.6263330106195653) q[10];
ry(-0.3238763279936228) q[11];
rz(0.2910502538971329) q[11];
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
ry(-0.45036319998997243) q[0];
rz(2.4587199399791215) q[0];
ry(0.07864388816571821) q[1];
rz(-1.466282287476937) q[1];
ry(-2.8854130714631827) q[2];
rz(0.7663561524067645) q[2];
ry(0.016449148892272447) q[3];
rz(2.4166038307693865) q[3];
ry(-0.0027562064960031663) q[4];
rz(2.625942750832847) q[4];
ry(3.1021402203428337) q[5];
rz(0.57763733727438) q[5];
ry(0.819426080279868) q[6];
rz(2.7874898630210803) q[6];
ry(-1.5912478282127784) q[7];
rz(0.014591074512368074) q[7];
ry(0.00019637064990406217) q[8];
rz(1.7978008590105965) q[8];
ry(2.69596749543843) q[9];
rz(-0.2959425467214043) q[9];
ry(-2.618880485947938) q[10];
rz(0.42832114948001665) q[10];
ry(1.8738609740067291) q[11];
rz(1.6907705366061956) q[11];
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
ry(-1.7301975781017855) q[0];
rz(-2.0185700850649018) q[0];
ry(0.05036375520167427) q[1];
rz(-2.68049177974166) q[1];
ry(2.2005768844293705) q[2];
rz(2.5368893067531455) q[2];
ry(0.6232762649059369) q[3];
rz(2.2575917526275897) q[3];
ry(-1.5296228816998412) q[4];
rz(2.09629586353788) q[4];
ry(2.9215088076484506) q[5];
rz(3.133774625764634) q[5];
ry(-3.0574166886239844) q[6];
rz(2.9623707967224737) q[6];
ry(-1.8845252171688929) q[7];
rz(0.00611088531831161) q[7];
ry(-2.7926309915878984) q[8];
rz(-1.2814077155274113) q[8];
ry(1.5559493300863947) q[9];
rz(0.7462928349041513) q[9];
ry(0.5965041282999433) q[10];
rz(-1.1014473397322897) q[10];
ry(-0.04884318328980441) q[11];
rz(-2.3471319513175377) q[11];
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
ry(-0.735272058479334) q[0];
rz(-0.34295825397597157) q[0];
ry(0.35650451287208185) q[1];
rz(-3.044697308466586) q[1];
ry(1.5582965809786833) q[2];
rz(-2.734487906562846) q[2];
ry(1.5659308633117401) q[3];
rz(-1.5350800785927214) q[3];
ry(-3.0924098615278814) q[4];
rz(1.8012648976469157) q[4];
ry(-0.016523300228574733) q[5];
rz(-0.6623917747028294) q[5];
ry(1.4173825156791438) q[6];
rz(3.0867923691812122) q[6];
ry(-0.7329449908610366) q[7];
rz(0.044558190736311856) q[7];
ry(0.0004919346219424245) q[8];
rz(1.7438713037891764) q[8];
ry(-3.0557637814952154) q[9];
rz(2.9355742377385794) q[9];
ry(1.9736937762275912) q[10];
rz(1.6752575528986346) q[10];
ry(0.15475011916306425) q[11];
rz(2.3131925005819687) q[11];
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
ry(-2.0630767353012898) q[0];
rz(-1.2445443298509398) q[0];
ry(1.126413520168394) q[1];
rz(0.8750506276545877) q[1];
ry(1.5804206556903708) q[2];
rz(-1.5891329439069513) q[2];
ry(1.5170991818741753) q[3];
rz(-1.4420325464025145) q[3];
ry(1.8129133719693247) q[4];
rz(0.8993205370447654) q[4];
ry(-2.630630791448108) q[5];
rz(2.0417630852130824) q[5];
ry(-0.21241038129300804) q[6];
rz(-0.7618681995463761) q[6];
ry(-2.3575962029896274) q[7];
rz(-3.118525986303252) q[7];
ry(-2.3600811264831387) q[8];
rz(2.988768654710022) q[8];
ry(0.7454214375446262) q[9];
rz(-1.530592313857007) q[9];
ry(0.6124805020663808) q[10];
rz(-0.23085619160863224) q[10];
ry(-0.10047551936259146) q[11];
rz(-2.6435499666210682) q[11];
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
ry(2.372599696077295) q[0];
rz(0.7311077572098768) q[0];
ry(-3.1407270092273096) q[1];
rz(-2.6513782025200987) q[1];
ry(0.5140794362458944) q[2];
rz(0.9227528219546581) q[2];
ry(0.0013390229332266301) q[3];
rz(2.054436830035476) q[3];
ry(-0.008210343414132737) q[4];
rz(2.393197415633696) q[4];
ry(0.008668856193893178) q[5];
rz(-2.1244599370611517) q[5];
ry(0.05508275276049978) q[6];
rz(-2.78024422853028) q[6];
ry(-0.4452003416593575) q[7];
rz(-1.0574379826272047) q[7];
ry(1.9912565057904903) q[8];
rz(-0.0033201183102976256) q[8];
ry(-2.4072834089626824) q[9];
rz(2.9191856785363037) q[9];
ry(1.4694861518602416) q[10];
rz(2.0511365061004363) q[10];
ry(-0.0013736886235928836) q[11];
rz(2.2917060668125204) q[11];
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
ry(-1.6566897381945374) q[0];
rz(0.8128695228509883) q[0];
ry(-2.3256131292741267) q[1];
rz(-2.8681891414382545) q[1];
ry(-0.007798701698153928) q[2];
rz(2.2264421213311065) q[2];
ry(3.1340755712793547) q[3];
rz(0.5866031359997224) q[3];
ry(1.4426153148752123) q[4];
rz(-1.6244017999087266) q[4];
ry(-0.31203864410985455) q[5];
rz(-1.8944387269694671) q[5];
ry(-2.5335708134823305) q[6];
rz(-0.18200638681631975) q[6];
ry(3.129919462739504) q[7];
rz(2.625101761056526) q[7];
ry(-1.1638916023970438) q[8];
rz(3.101261073974134) q[8];
ry(3.1251525618956837) q[9];
rz(0.08192224485892474) q[9];
ry(0.796890191309398) q[10];
rz(-2.5452862534172978) q[10];
ry(0.014539115579228525) q[11];
rz(-3.0106218125767343) q[11];
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
ry(-0.5120782811512736) q[0];
rz(0.1826544004770882) q[0];
ry(-2.265158687179845) q[1];
rz(-0.28409979849350214) q[1];
ry(0.6501698752267124) q[2];
rz(-0.8249042216245063) q[2];
ry(-2.9944545993721277) q[3];
rz(-0.05812862513482208) q[3];
ry(3.134187997348363) q[4];
rz(-1.908666161624288) q[4];
ry(-0.026993612497546057) q[5];
rz(-0.8659495816348413) q[5];
ry(-0.5297977379117594) q[6];
rz(2.0922913987296714) q[6];
ry(3.0896660463866894) q[7];
rz(2.097548007310647) q[7];
ry(0.428944076237487) q[8];
rz(1.635010600204938) q[8];
ry(-2.1345395155161238) q[9];
rz(0.009501650511444866) q[9];
ry(2.1277042193089652) q[10];
rz(0.15193337128714718) q[10];
ry(1.8640805811378829) q[11];
rz(-2.619813563912229) q[11];
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
ry(3.13529930723609) q[0];
rz(-0.6324877788404457) q[0];
ry(-0.3242295154131519) q[1];
rz(1.1961022685147267) q[1];
ry(-3.1141843231753104) q[2];
rz(-2.73168544186149) q[2];
ry(1.5473170436237893) q[3];
rz(-0.43714482978801217) q[3];
ry(-3.1089570082048232) q[4];
rz(2.5650792082592413) q[4];
ry(1.547027220047529) q[5];
rz(-1.4686400144087997) q[5];
ry(-2.5919743402621727) q[6];
rz(-2.352020393497604) q[6];
ry(1.5875790471207374) q[7];
rz(0.06244768464712611) q[7];
ry(0.9904033168135374) q[8];
rz(-0.07879134784062529) q[8];
ry(1.566262667124616) q[9];
rz(-2.5353300552530804) q[9];
ry(-2.511293522648502) q[10];
rz(-3.015072897229719) q[10];
ry(2.9852596554000397) q[11];
rz(1.2073230140191562) q[11];
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
ry(-0.2899295848229105) q[0];
rz(0.7635085523658764) q[0];
ry(-2.214131591041366) q[1];
rz(2.2188391206804714) q[1];
ry(0.021815855610446546) q[2];
rz(-1.2614754863926647) q[2];
ry(2.988275764231991) q[3];
rz(2.7701100384205937) q[3];
ry(-3.0778275811609115) q[4];
rz(1.5897670431495288) q[4];
ry(0.09323307421007154) q[5];
rz(-1.7525554348796935) q[5];
ry(-0.04241032017918827) q[6];
rz(3.0915910430931617) q[6];
ry(-3.0947071132351236) q[7];
rz(-1.3557747325874099) q[7];
ry(3.087888852780288) q[8];
rz(3.1230625277961463) q[8];
ry(-3.139910205582278) q[9];
rz(-0.9650025048593216) q[9];
ry(-1.558593249172367) q[10];
rz(-1.5739316207724168) q[10];
ry(-1.85941940838008) q[11];
rz(-2.4695185098138235) q[11];
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
ry(-0.0035792069384594782) q[0];
rz(-1.59313221444575) q[0];
ry(1.6101681340326124) q[1];
rz(2.778604502886139) q[1];
ry(1.5307187699702598) q[2];
rz(2.731616753818823) q[2];
ry(-1.6245177679386371) q[3];
rz(-2.9820664519339815) q[3];
ry(1.568861530331728) q[4];
rz(0.11179317883522179) q[4];
ry(1.500527933995584) q[5];
rz(1.668134025002953) q[5];
ry(-1.4360846917321082) q[6];
rz(-1.584281695451164) q[6];
ry(-0.020201091492041634) q[7];
rz(-0.27708318751918193) q[7];
ry(-0.584363038385041) q[8];
rz(-1.1695868835248813) q[8];
ry(1.5736732783063005) q[9];
rz(1.6584769352706188) q[9];
ry(-1.5709218166782797) q[10];
rz(1.6110379454613233) q[10];
ry(3.1398037798513796) q[11];
rz(-2.686534915698556) q[11];