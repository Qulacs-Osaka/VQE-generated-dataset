OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.9332074824673406) q[0];
rz(-1.4942981902609451) q[0];
ry(0.17452094511681437) q[1];
rz(0.723607797908703) q[1];
ry(-3.1378211841601304) q[2];
rz(-0.3158926133502993) q[2];
ry(-0.0015914731503974269) q[3];
rz(-1.376373001282797) q[3];
ry(0.01083176892997173) q[4];
rz(2.163226554284156) q[4];
ry(-0.0008781769535337601) q[5];
rz(-2.0599172098696936) q[5];
ry(1.625697511641321) q[6];
rz(2.7896834634298076) q[6];
ry(-1.5695738614401016) q[7];
rz(-1.3689745639016626) q[7];
ry(1.4661045583661578) q[8];
rz(-1.5698431934733756) q[8];
ry(-1.5710086498511453) q[9];
rz(-1.5806741442019563) q[9];
ry(-0.00240876659459488) q[10];
rz(0.19573272156406762) q[10];
ry(0.2409284656356112) q[11];
rz(0.011048907637336444) q[11];
ry(-0.017680338144731925) q[12];
rz(-2.386253380353275) q[12];
ry(-0.001960895673691631) q[13];
rz(0.9588646913009184) q[13];
ry(1.5685580932403747) q[14];
rz(3.130185780066387) q[14];
ry(-1.6234902281904269) q[15];
rz(-0.880507516102928) q[15];
ry(3.0982047109820057) q[16];
rz(0.11192938682211029) q[16];
ry(3.139540980542235) q[17];
rz(2.045086697514518) q[17];
ry(1.8292555145252107) q[18];
rz(0.028875362158931498) q[18];
ry(1.8447776517239816) q[19];
rz(-2.887339908864292) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.4750692685461197) q[0];
rz(-2.534325154518341) q[0];
ry(1.7763192788062074) q[1];
rz(-1.7632682263103538) q[1];
ry(3.1404563209795895) q[2];
rz(-2.4475786646893147) q[2];
ry(-0.0011569852741555309) q[3];
rz(-1.35645587225755) q[3];
ry(-1.172848970612006) q[4];
rz(3.135161173289669) q[4];
ry(-2.858023573613277) q[5];
rz(-0.035743837634934994) q[5];
ry(3.133437841439704) q[6];
rz(-1.9165357803047316) q[6];
ry(-3.0832143265270573) q[7];
rz(-1.2930130186370627) q[7];
ry(1.5717798805450345) q[8];
rz(3.1135424810261574) q[8];
ry(1.9746049167340045) q[9];
rz(3.109737646468563) q[9];
ry(0.06648386052692595) q[10];
rz(-1.573084535369607) q[10];
ry(1.570368301038224) q[11];
rz(-1.5775109861180843) q[11];
ry(0.2615854205566102) q[12];
rz(-2.600222596825436) q[12];
ry(3.1100300309035176) q[13];
rz(-0.3387902861295125) q[13];
ry(0.7176196511810486) q[14];
rz(0.6973421100126842) q[14];
ry(2.158667265974134) q[15];
rz(-0.551083703732603) q[15];
ry(-3.0847244487053906) q[16];
rz(0.6622750296932172) q[16];
ry(0.2750107036057461) q[17];
rz(-0.028406830952887322) q[17];
ry(0.14193459109178352) q[18];
rz(-0.27995550017194143) q[18];
ry(-1.3975039865626506) q[19];
rz(-1.6917328497212019) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.26513995037758287) q[0];
rz(2.5530542558192715) q[0];
ry(-2.445107984619987) q[1];
rz(-2.050419132160868) q[1];
ry(-1.5635673873982567) q[2];
rz(-0.5538388734723034) q[2];
ry(1.5806497319561843) q[3];
rz(-1.4229434901441629) q[3];
ry(1.5656964340313095) q[4];
rz(2.9692647919776767) q[4];
ry(1.572007409028341) q[5];
rz(0.8486054633880613) q[5];
ry(-1.5791609358576886) q[6];
rz(-3.1338968601494606) q[6];
ry(1.5694604583924936) q[7];
rz(0.008574442245683562) q[7];
ry(1.5809431379902694) q[8];
rz(-1.6608003163713596) q[8];
ry(-1.5704176333271862) q[9];
rz(1.571484766562162) q[9];
ry(-1.543528052890535) q[10];
rz(1.5118504756237385) q[10];
ry(-1.572570902897953) q[11];
rz(0.45939295849220724) q[11];
ry(3.141349489860262) q[12];
rz(-2.6194146794539472) q[12];
ry(-3.141575460495592) q[13];
rz(2.0872009624008014) q[13];
ry(-1.5757967736398804) q[14];
rz(-1.6595454614654557) q[14];
ry(1.3198981380105268) q[15];
rz(-1.6771590320682828) q[15];
ry(0.9763115846573945) q[16];
rz(2.149380628884664) q[16];
ry(-1.5304054909338152) q[17];
rz(1.5693934962216458) q[17];
ry(3.0126654103511497) q[18];
rz(0.6374412951326462) q[18];
ry(-2.010317612353382) q[19];
rz(1.4018709770470679) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.672453148428099) q[0];
rz(-1.4541935949940987) q[0];
ry(2.976899667862316) q[1];
rz(-1.0921375597064458) q[1];
ry(6.747232476556064e-05) q[2];
rz(-1.1754830599305557) q[2];
ry(0.0003233205859700661) q[3];
rz(-0.19358744101430062) q[3];
ry(1.756883636554315e-05) q[4];
rz(-2.945074137605901) q[4];
ry(-0.0002437698939630195) q[5];
rz(-0.8480727222135133) q[5];
ry(-1.5706159720095336) q[6];
rz(-1.5714284876231286) q[6];
ry(-1.5706812587299535) q[7];
rz(-1.6660194516984799) q[7];
ry(3.141504371876324) q[8];
rz(-2.195363996807958) q[8];
ry(-2.917800088555993) q[9];
rz(-1.486955415924288) q[9];
ry(3.102020462626599) q[10];
rz(1.9034394534689203) q[10];
ry(3.1378254918780466) q[11];
rz(0.7820671181910859) q[11];
ry(3.595708788413817e-05) q[12];
rz(1.3859039290142319) q[12];
ry(3.141572931619677) q[13];
rz(2.849167936764141) q[13];
ry(1.5721435739259202) q[14];
rz(-0.2955596197092562) q[14];
ry(1.5671933715206618) q[15];
rz(-0.013381405858322268) q[15];
ry(-3.1010915203094607) q[16];
rz(-2.2104339560948203) q[16];
ry(-2.1303409780712714) q[17];
rz(-3.1379604133947936) q[17];
ry(3.141048506970222) q[18];
rz(2.40969912429674) q[18];
ry(0.05163206317298475) q[19];
rz(2.205220785988102) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.9476799872672874) q[0];
rz(0.7159058395393801) q[0];
ry(1.0517973495876918) q[1];
rz(-1.1857235961699015) q[1];
ry(0.10821278456421422) q[2];
rz(0.25155685002137496) q[2];
ry(1.6137768971316253) q[3];
rz(1.6946300890980095) q[3];
ry(0.4143214890436625) q[4];
rz(1.6320911931322248) q[4];
ry(-1.4593732676978517) q[5];
rz(1.5003170161193544) q[5];
ry(-1.6531081141179031) q[6];
rz(-2.732494158399513) q[6];
ry(3.13017913616151) q[7];
rz(-0.16708179122717762) q[7];
ry(-3.0428321053733502) q[8];
rz(-1.6479809462233188) q[8];
ry(-0.3314640114019091) q[9];
rz(-2.7734034888031944) q[9];
ry(3.117751522621119) q[10];
rz(1.918602254637159) q[10];
ry(3.137104163469105) q[11];
rz(-3.1357593161332384) q[11];
ry(3.139064596623826) q[12];
rz(1.5190829955985778) q[12];
ry(0.00014572764290817956) q[13];
rz(1.0646135849711162) q[13];
ry(2.698214189265354) q[14];
rz(-0.32622136639663146) q[14];
ry(0.5633223910632807) q[15];
rz(-3.123752779142454) q[15];
ry(-0.12303102703280633) q[16];
rz(-2.9809525572477353) q[16];
ry(-1.465602001365692) q[17];
rz(2.220754434092102) q[17];
ry(3.0676383979344184) q[18];
rz(1.9274259158474367) q[18];
ry(-2.952993577103938) q[19];
rz(-1.920907658384886) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.076492604465863) q[0];
rz(2.6861871903536136) q[0];
ry(1.9368707711162747) q[1];
rz(0.638372066348807) q[1];
ry(-3.129764456217392) q[2];
rz(0.007094708004430572) q[2];
ry(0.17354592816281933) q[3];
rz(1.459467977545659) q[3];
ry(2.880681887894667) q[4];
rz(1.4638189825709889) q[4];
ry(0.03276053870538294) q[5];
rz(-2.612267915398521) q[5];
ry(0.0002995718050007312) q[6];
rz(-0.4139425562156065) q[6];
ry(3.140706445929934) q[7];
rz(3.057790989684038) q[7];
ry(-7.293109801231878e-05) q[8];
rz(-0.4105454330520341) q[8];
ry(0.003756569783882391) q[9];
rz(1.3830758188689631) q[9];
ry(3.107077405017932) q[10];
rz(1.5207389207204636) q[10];
ry(0.001647333777554562) q[11];
rz(-1.2812398020565985) q[11];
ry(1.5709815034273147) q[12];
rz(-1.6458327590604123) q[12];
ry(-1.574083531237303) q[13];
rz(0.821350603751818) q[13];
ry(-1.6026218580909064) q[14];
rz(-1.7822302057317456) q[14];
ry(-1.5700579341929388) q[15];
rz(1.8788699048016941) q[15];
ry(2.962036636903626) q[16];
rz(-2.072085886963799) q[16];
ry(-2.821196748514284) q[17];
rz(-1.227478969372757) q[17];
ry(0.07942011202493691) q[18];
rz(1.6090701960473597) q[18];
ry(-3.1261183680250983) q[19];
rz(-0.38126369486069756) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.081092699541962) q[0];
rz(-2.4337162140948045) q[0];
ry(0.7590950059003694) q[1];
rz(-0.6816425429146786) q[1];
ry(2.840751881447339) q[2];
rz(2.149625343483286) q[2];
ry(1.0590443747671818) q[3];
rz(-2.2048911689177784) q[3];
ry(1.8766241472240746) q[4];
rz(-0.015001876353265333) q[4];
ry(-1.5797918479454918) q[5];
rz(-0.038460928516029995) q[5];
ry(-1.5713867714899044) q[6];
rz(-3.1084511940312995) q[6];
ry(-1.569801784203566) q[7];
rz(-2.9608032941612663) q[7];
ry(1.9173965385149607) q[8];
rz(-2.322205881990428) q[8];
ry(-2.282360575762857) q[9];
rz(0.43479272058834256) q[9];
ry(-1.6973317183030447) q[10];
rz(0.13812632161789648) q[10];
ry(1.6182991272683154) q[11];
rz(-0.546685334980164) q[11];
ry(-1.5286548037356777) q[12];
rz(-1.0899971225115692) q[12];
ry(1.5516640106764354) q[13];
rz(0.7825171911293825) q[13];
ry(3.141543534703904) q[14];
rz(-1.3896527178100193) q[14];
ry(-0.00016922669412000602) q[15];
rz(1.196178628698181) q[15];
ry(0.7331405949959207) q[16];
rz(-3.101910396586548) q[16];
ry(1.1232415293154316) q[17];
rz(-0.4030353826707632) q[17];
ry(-0.12072402534059723) q[18];
rz(-0.045856359803067015) q[18];
ry(-0.20139544225205738) q[19];
rz(-0.3588765132037448) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5319581777206734) q[0];
rz(2.9074392703617558) q[0];
ry(-0.5211099283952798) q[1];
rz(3.1393287789945488) q[1];
ry(3.140157919887403) q[2];
rz(2.267868891287603) q[2];
ry(-3.139811041928869) q[3];
rz(-2.237498139458591) q[3];
ry(1.5654741122583609) q[4];
rz(-0.01730056884310602) q[4];
ry(1.4838783107928233) q[5];
rz(-3.1018091589996892) q[5];
ry(3.135417739578621) q[6];
rz(-1.6724438045204817) q[6];
ry(-0.006372728210849665) q[7];
rz(-1.4385060574975457) q[7];
ry(0.002724238663810219) q[8];
rz(2.677047988005515) q[8];
ry(-3.1405225344457355) q[9];
rz(-3.125018318531508) q[9];
ry(-0.02705669972738267) q[10];
rz(3.0303117426283523) q[10];
ry(-0.0010061107323112392) q[11];
rz(-2.597350984731425) q[11];
ry(-0.0050565062749839814) q[12];
rz(-0.47791778493799164) q[12];
ry(0.004913144148377388) q[13];
rz(-2.3359770704821825) q[13];
ry(3.1415232930966828) q[14];
rz(-2.648530749239448) q[14];
ry(3.141540840840702) q[15];
rz(-0.9147823987066518) q[15];
ry(0.5670922485558743) q[16];
rz(2.687098813718895) q[16];
ry(-0.8644087868260009) q[17];
rz(-2.7733852203418317) q[17];
ry(-3.075025395800376) q[18];
rz(-2.57416610915837) q[18];
ry(0.23453339822742958) q[19];
rz(0.6521078056398745) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.010705120418527336) q[0];
rz(-2.2695731807535764) q[0];
ry(1.6349455554368388) q[1];
rz(-0.3007675516726381) q[1];
ry(-1.971129371327219) q[2];
rz(3.11556809143199) q[2];
ry(1.1714504855542833) q[3];
rz(1.5676474058263377) q[3];
ry(-1.8576229327764784) q[4];
rz(-2.62078151015292) q[4];
ry(1.6245917639290353) q[5];
rz(-1.4676883762908517) q[5];
ry(-3.137671027258057) q[6];
rz(3.073855556944832) q[6];
ry(-3.1286716447957206) q[7];
rz(-2.629697597283686) q[7];
ry(-0.09329809969597741) q[8];
rz(-1.8192887476514708) q[8];
ry(2.287318531151025) q[9];
rz(1.5480865946562137) q[9];
ry(1.6987746986532732) q[10];
rz(3.0193914974547424) q[10];
ry(-1.6190821403163786) q[11];
rz(-0.9230459983779493) q[11];
ry(-1.498921830794339) q[12];
rz(1.552639274732627) q[12];
ry(0.8295017807060852) q[13];
rz(-1.5667379415479596) q[13];
ry(0.00041094174958988816) q[14];
rz(2.390327793954822) q[14];
ry(3.1412553523369082) q[15];
rz(2.48538252858774) q[15];
ry(-0.21359748798474748) q[16];
rz(-2.9485377012822367) q[16];
ry(-1.2631025875842221) q[17];
rz(0.2842549078186946) q[17];
ry(-1.5113716964397583) q[18];
rz(1.6368533526228308) q[18];
ry(-3.0611713085810894) q[19];
rz(1.7206702517851813) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.18328074062369543) q[0];
rz(-2.469897700494186) q[0];
ry(2.123756306191414) q[1];
rz(-0.12271584822307607) q[1];
ry(1.5568330543322004) q[2];
rz(-2.957844358159812) q[2];
ry(1.1583119728379851) q[3];
rz(-1.0497658948124322) q[3];
ry(-1.5818675683140395) q[4];
rz(-1.638519333892873) q[4];
ry(-1.5887956873799114) q[5];
rz(1.251322059722124) q[5];
ry(3.1356459121735942) q[6];
rz(-1.5043346046316166) q[6];
ry(-0.00035810951844723226) q[7];
rz(1.3706889012872878) q[7];
ry(1.5696271827034112) q[8];
rz(-2.8125734663199196) q[8];
ry(1.5701061158796603) q[9];
rz(-2.0441913054800382) q[9];
ry(3.1351706744108245) q[10];
rz(-1.7028881907767548) q[10];
ry(-3.139881586480795) q[11];
rz(2.185832930466834) q[11];
ry(1.8236443162513511) q[12];
rz(0.9344623819961669) q[12];
ry(-1.5794579497444015) q[13];
rz(-1.9074262251737955) q[13];
ry(-0.002991343187805917) q[14];
rz(-3.070225058503213) q[14];
ry(-3.0814028212903226) q[15];
rz(1.5251567119839406) q[15];
ry(2.9147150514235602) q[16];
rz(-0.5519163678455294) q[16];
ry(2.5331215919148558) q[17];
rz(1.6076429852601364) q[17];
ry(1.3928063272467206) q[18];
rz(-1.5520903345618815) q[18];
ry(0.20145763609263412) q[19];
rz(-1.4935267167099862) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.6479219525386934) q[0];
rz(-2.8319766614235076) q[0];
ry(1.627222657582292) q[1];
rz(3.0779984125341926) q[1];
ry(3.141541502343974) q[2];
rz(-0.987963745336212) q[2];
ry(-3.1411416283558444) q[3];
rz(-2.6258219917380337) q[3];
ry(0.024564188860648706) q[4];
rz(-1.5388224837483704) q[4];
ry(-3.1303100751639787) q[5];
rz(1.569280819315018) q[5];
ry(-1.5683206464440687) q[6];
rz(-1.8641438526814196) q[6];
ry(1.5704829659767308) q[7];
rz(0.08049695148399039) q[7];
ry(0.003806430583571764) q[8];
rz(3.0184187470467743) q[8];
ry(1.7577898105645098) q[9];
rz(1.4745449213677884) q[9];
ry(1.570090664929187) q[10];
rz(-1.5626992693191122) q[10];
ry(1.5602128120632912) q[11];
rz(0.616286430969569) q[11];
ry(-3.1312359863568124) q[12];
rz(-2.2955095564828296) q[12];
ry(3.1368537334913422) q[13];
rz(1.355732227515718) q[13];
ry(-0.009520627095941264) q[14];
rz(-2.745884999843849) q[14];
ry(-0.007140964542055734) q[15];
rz(0.06826975393592694) q[15];
ry(-1.7004734304661782) q[16];
rz(1.5410557229947628) q[16];
ry(-3.0452692875287983) q[17];
rz(0.8845846316462082) q[17];
ry(1.384416135309436) q[18];
rz(-2.3507320267286427) q[18];
ry(-0.28098652981506334) q[19];
rz(0.9569653277342844) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.161595835685982) q[0];
rz(0.24018106822704283) q[0];
ry(1.7593412591813873) q[1];
rz(1.8694947896830303) q[1];
ry(1.5708510606232435) q[2];
rz(-1.984086495643318) q[2];
ry(1.5666995987795032) q[3];
rz(1.2385781446247524) q[3];
ry(1.5749018294642365) q[4];
rz(2.9589130258299843) q[4];
ry(-1.5679828118113395) q[5];
rz(2.059929424109247) q[5];
ry(-3.1162353616467913) q[6];
rz(1.2679025874447136) q[6];
ry(3.1171014376181856) q[7];
rz(-1.5008519354067031) q[7];
ry(-3.0652616754040753) q[8];
rz(1.743749861343189) q[8];
ry(1.6365284907830582) q[9];
rz(3.1157480414764676) q[9];
ry(2.011475601506164) q[10];
rz(3.135186989929652) q[10];
ry(-0.0012967335884818444) q[11];
rz(-0.5552479261749419) q[11];
ry(-1.5645640982581446) q[12];
rz(-0.2072351438497808) q[12];
ry(-1.5672688992566228) q[13];
rz(1.8362816231071513) q[13];
ry(0.362922874263476) q[14];
rz(-0.6554432408813534) q[14];
ry(1.4589861838024845) q[15];
rz(2.882254897755665) q[15];
ry(0.7371124970360317) q[16];
rz(-2.521879087568863) q[16];
ry(1.0532604751563452) q[17];
rz(1.419272780488276) q[17];
ry(-3.1266042113943397) q[18];
rz(-2.2267660157213136) q[18];
ry(-0.7149572011445642) q[19];
rz(2.707542881137709) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.8476467309161233) q[0];
rz(1.6494369862543339) q[0];
ry(-3.1171328448381788) q[1];
rz(2.2378255460146965) q[1];
ry(-3.1297744146147033) q[2];
rz(1.4364897658739577) q[2];
ry(-0.004882513238828988) q[3];
rz(1.763673482253994) q[3];
ry(1.5702613630543645) q[4];
rz(3.139365235763401) q[4];
ry(1.576287550755902) q[5];
rz(-3.1408474212681408) q[5];
ry(0.057765000843038844) q[6];
rz(-1.1106195129663208) q[6];
ry(1.8897514753275542) q[7];
rz(-1.232594291402382) q[7];
ry(0.009033269251939896) q[8];
rz(3.053973801098501) q[8];
ry(1.8489898185317464) q[9];
rz(1.7759873238977486) q[9];
ry(-0.2333570486535104) q[10];
rz(0.46185919481353116) q[10];
ry(0.000821388763284166) q[11];
rz(-0.15304295655996558) q[11];
ry(-0.00026796865480889184) q[12];
rz(-1.5445456045678538) q[12];
ry(0.000767321533740688) q[13];
rz(-2.096099239772127) q[13];
ry(-1.1162523694707716e-05) q[14];
rz(2.6183734833481207) q[14];
ry(-3.141564930571531) q[15];
rz(1.9579501977858504) q[15];
ry(-0.007470549227758028) q[16];
rz(-2.0687682290309906) q[16];
ry(-1.5668438521893029) q[17];
rz(-1.4579749799992312) q[17];
ry(-1.0701647433096175) q[18];
rz(-0.4261123250606823) q[18];
ry(-3.0416307642414564) q[19];
rz(-2.5691961729957717) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.887137897798027) q[0];
rz(1.817509306615776) q[0];
ry(1.235510400509379) q[1];
rz(-1.5920956960355994) q[1];
ry(3.1393328487141923) q[2];
rz(0.278250487115225) q[2];
ry(-0.0007995617930035238) q[3];
rz(-3.003090522874695) q[3];
ry(-1.5711677291791357) q[4];
rz(-3.1415916583157575) q[4];
ry(1.5711985573105325) q[5];
rz(3.1414008491843375) q[5];
ry(-3.1410860010777992) q[6];
rz(-2.670907306203073) q[6];
ry(0.0006015237594061399) q[7];
rz(-0.0572420219036367) q[7];
ry(1.5721790649799192) q[8];
rz(-3.135202177551678) q[8];
ry(-1.5726224523723462) q[9];
rz(-0.02675742333404562) q[9];
ry(2.631358535133574) q[10];
rz(1.8388658610720547) q[10];
ry(-3.1140650459325725) q[11];
rz(0.6005239639112947) q[11];
ry(1.2014437633576405) q[12];
rz(-2.0007700794406267) q[12];
ry(-0.0017962489830853912) q[13];
rz(1.1576510238131528) q[13];
ry(-1.1135533189365363) q[14];
rz(0.742124779259715) q[14];
ry(1.8225501273319553) q[15];
rz(-3.0086531558948746) q[15];
ry(-3.1116047118034214) q[16];
rz(-1.9934941538901227) q[16];
ry(-1.6151547764109344) q[17];
rz(-3.136446552593974) q[17];
ry(-2.78549350943034) q[18];
rz(-2.0331509877230665) q[18];
ry(1.4746070948672558) q[19];
rz(-0.18047108704566012) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.950066021632618) q[0];
rz(-2.7098131556596003) q[0];
ry(2.780129497147821) q[1];
rz(0.7363693282061653) q[1];
ry(1.570563002993194) q[2];
rz(1.5476369140261976) q[2];
ry(1.5707240932586481) q[3];
rz(-3.072482456261457) q[3];
ry(-1.5706723590166716) q[4];
rz(2.6277742476168937) q[4];
ry(1.57124812518828) q[5];
rz(1.3985523815948575) q[5];
ry(0.02373751290602567) q[6];
rz(-0.5526839418893842) q[6];
ry(-0.012638925424338865) q[7];
rz(3.036597701188204) q[7];
ry(-1.5486459307319742) q[8];
rz(1.6017911095237118) q[8];
ry(-1.5713829632463785) q[9];
rz(2.9595110265257567) q[9];
ry(-0.0013060546982103145) q[10];
rz(1.703636497443699) q[10];
ry(-0.0006647661933777054) q[11];
rz(0.90487496476125) q[11];
ry(3.134766444553022) q[12];
rz(0.6549276946562504) q[12];
ry(0.009159525101633648) q[13];
rz(-1.7220939200536112) q[13];
ry(0.0005543609038020326) q[14];
rz(0.7000012697825403) q[14];
ry(-3.1413555412058747) q[15];
rz(-1.602815902697655) q[15];
ry(-3.1408437634513016) q[16];
rz(-2.9646089968364313) q[16];
ry(0.004056278718057292) q[17];
rz(0.8283431081988091) q[17];
ry(-1.3339821981429472) q[18];
rz(0.4275077517698654) q[18];
ry(-1.3841230405570943) q[19];
rz(2.301909032987922) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1395531094498894) q[0];
rz(-2.708548754790116) q[0];
ry(3.136991915976023) q[1];
rz(-2.35967829761866) q[1];
ry(1.5816734608122431) q[2];
rz(3.1367755421755232) q[2];
ry(1.5392091449202248) q[3];
rz(-1.7273122568907233) q[3];
ry(-1.518496226831216) q[4];
rz(-0.05216235205762753) q[4];
ry(-1.405459900984373) q[5];
rz(-3.137013749783386) q[5];
ry(-0.003183469284970975) q[6];
rz(-1.0478393994351096) q[6];
ry(3.1369831258009446) q[7];
rz(1.7485453136895888) q[7];
ry(-1.7810306787851067) q[8];
rz(1.5954665251275086) q[8];
ry(0.015809445145242407) q[9];
rz(-1.391987227676304) q[9];
ry(-1.5889555995921039) q[10];
rz(-1.5774331027581143) q[10];
ry(1.5681046552210534) q[11];
rz(0.003219777730323213) q[11];
ry(0.3902013215759901) q[12];
rz(-1.105412256390213) q[12];
ry(0.001522758582753525) q[13];
rz(0.8316445909927639) q[13];
ry(1.1978599714491298) q[14];
rz(0.47625951537022215) q[14];
ry(2.088093316315439) q[15];
rz(-0.2922868253528277) q[15];
ry(-1.4890689105535437) q[16];
rz(-0.030959824575203857) q[16];
ry(1.7646597267010644) q[17];
rz(0.5264703350164436) q[17];
ry(-1.8986322040540242) q[18];
rz(2.264618766419191) q[18];
ry(-1.7977761287406302) q[19];
rz(0.21400264786678633) q[19];