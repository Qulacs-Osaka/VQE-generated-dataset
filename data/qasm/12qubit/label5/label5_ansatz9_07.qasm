OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.41448273530276364) q[0];
ry(-1.340490140977264) q[1];
cx q[0],q[1];
ry(-1.6101287829853408) q[0];
ry(0.8307352631904801) q[1];
cx q[0],q[1];
ry(-1.8432375795430849) q[2];
ry(-0.6820368302440476) q[3];
cx q[2],q[3];
ry(0.3006171470379595) q[2];
ry(-0.22383034115329767) q[3];
cx q[2],q[3];
ry(-1.258719068556913) q[4];
ry(-3.0862456968673437) q[5];
cx q[4],q[5];
ry(2.7630922582483737) q[4];
ry(1.8566285110087) q[5];
cx q[4],q[5];
ry(0.8335983580814859) q[6];
ry(-2.5205977254606315) q[7];
cx q[6],q[7];
ry(0.6747687865232832) q[6];
ry(-2.5589776100788657) q[7];
cx q[6],q[7];
ry(0.7629092792065726) q[8];
ry(-1.6654569153989751) q[9];
cx q[8],q[9];
ry(2.12085853482729) q[8];
ry(0.9096154288324924) q[9];
cx q[8],q[9];
ry(-1.6816976727809132) q[10];
ry(-0.8656052750534927) q[11];
cx q[10],q[11];
ry(-2.7753016649375444) q[10];
ry(-2.7748716798092574) q[11];
cx q[10],q[11];
ry(-1.74255882456255) q[0];
ry(-1.0455661150171827) q[2];
cx q[0],q[2];
ry(0.16882275627334928) q[0];
ry(-1.4219050377027629) q[2];
cx q[0],q[2];
ry(1.7997209625791648) q[2];
ry(1.3623829334956197) q[4];
cx q[2],q[4];
ry(-1.1233684160038493) q[2];
ry(0.36449153904889053) q[4];
cx q[2],q[4];
ry(1.70656391042128) q[4];
ry(-1.8651851062005225) q[6];
cx q[4],q[6];
ry(2.5325041010779454) q[4];
ry(1.5258497083875855) q[6];
cx q[4],q[6];
ry(1.110247350106933) q[6];
ry(2.645587060623814) q[8];
cx q[6],q[8];
ry(-1.264530126264754) q[6];
ry(2.643245610296172) q[8];
cx q[6],q[8];
ry(1.9951761227708658) q[8];
ry(0.4498870558682944) q[10];
cx q[8],q[10];
ry(-0.7970688227472388) q[8];
ry(0.23045563637649638) q[10];
cx q[8],q[10];
ry(2.117987859211664) q[1];
ry(1.7612486840837052) q[3];
cx q[1],q[3];
ry(0.4772332273063679) q[1];
ry(-0.5596972926021673) q[3];
cx q[1],q[3];
ry(1.2123870461041157) q[3];
ry(2.419294830312316) q[5];
cx q[3],q[5];
ry(-2.4603026249470688) q[3];
ry(-0.6314050203584407) q[5];
cx q[3],q[5];
ry(2.0080933029014387) q[5];
ry(-2.3270369973228644) q[7];
cx q[5],q[7];
ry(1.1793578064287153) q[5];
ry(0.7459662758322336) q[7];
cx q[5],q[7];
ry(-0.7922420607527804) q[7];
ry(-2.0455660319245155) q[9];
cx q[7],q[9];
ry(1.3286080787200865) q[7];
ry(-1.8865229098320382) q[9];
cx q[7],q[9];
ry(-1.4781897357458984) q[9];
ry(2.001921910289848) q[11];
cx q[9],q[11];
ry(1.3348209249929965) q[9];
ry(-0.08407169837955195) q[11];
cx q[9],q[11];
ry(0.2776095346332908) q[0];
ry(2.027315619068229) q[3];
cx q[0],q[3];
ry(-2.302752006665351) q[0];
ry(-2.4395918839560067) q[3];
cx q[0],q[3];
ry(0.9896606485088969) q[1];
ry(2.607463961538396) q[2];
cx q[1],q[2];
ry(-2.7785444146533727) q[1];
ry(1.190254423929534) q[2];
cx q[1],q[2];
ry(-1.4849236390012495) q[2];
ry(1.6785049143341215) q[5];
cx q[2],q[5];
ry(-0.8441190217098331) q[2];
ry(1.2733721815332195) q[5];
cx q[2],q[5];
ry(0.7086305989482939) q[3];
ry(-0.90303806554369) q[4];
cx q[3],q[4];
ry(-0.6820939360120831) q[3];
ry(-1.5400538433704716) q[4];
cx q[3],q[4];
ry(3.0912291931493736) q[4];
ry(0.8236030037315218) q[7];
cx q[4],q[7];
ry(-1.0966245656781224) q[4];
ry(1.9188069614655687) q[7];
cx q[4],q[7];
ry(1.023014010084406) q[5];
ry(-1.8363936445428264) q[6];
cx q[5],q[6];
ry(1.977918631490463) q[5];
ry(1.480524727419305) q[6];
cx q[5],q[6];
ry(1.8090883687414245) q[6];
ry(1.5456473388005065) q[9];
cx q[6],q[9];
ry(-2.3082669341482904) q[6];
ry(1.3068852215865125) q[9];
cx q[6],q[9];
ry(-1.224204006358419) q[7];
ry(0.6539750495070594) q[8];
cx q[7],q[8];
ry(-0.9994765040428848) q[7];
ry(-0.4420302439992003) q[8];
cx q[7],q[8];
ry(0.7641161503186638) q[8];
ry(1.205624286553456) q[11];
cx q[8],q[11];
ry(-2.78460639864501) q[8];
ry(-2.6550540612958518) q[11];
cx q[8],q[11];
ry(2.326360214342657) q[9];
ry(2.1519932881626174) q[10];
cx q[9],q[10];
ry(1.5908301800512434) q[9];
ry(1.40676315964416) q[10];
cx q[9],q[10];
ry(-1.8337152748599443) q[0];
ry(0.7484422955137683) q[1];
cx q[0],q[1];
ry(2.9167655749364307) q[0];
ry(-1.1786663414689422) q[1];
cx q[0],q[1];
ry(0.3942601679694265) q[2];
ry(2.563409015419902) q[3];
cx q[2],q[3];
ry(0.4360885449140323) q[2];
ry(2.7588644733441225) q[3];
cx q[2],q[3];
ry(0.43729603298056435) q[4];
ry(-1.7217163291694924) q[5];
cx q[4],q[5];
ry(0.16648113847787016) q[4];
ry(-1.190297269524424) q[5];
cx q[4],q[5];
ry(1.5302377637310043) q[6];
ry(-3.129502324383938) q[7];
cx q[6],q[7];
ry(2.31433984585345) q[6];
ry(1.911291832397436) q[7];
cx q[6],q[7];
ry(-2.349743221755044) q[8];
ry(2.5838822679563997) q[9];
cx q[8],q[9];
ry(2.897029012248755) q[8];
ry(2.63752278889502) q[9];
cx q[8],q[9];
ry(0.1797771011441826) q[10];
ry(2.7252762320278556) q[11];
cx q[10],q[11];
ry(-0.8449278008985744) q[10];
ry(2.9224454860080598) q[11];
cx q[10],q[11];
ry(-1.7780705764270115) q[0];
ry(1.1495493811008313) q[2];
cx q[0],q[2];
ry(-1.0937769327235276) q[0];
ry(-2.435709700710388) q[2];
cx q[0],q[2];
ry(-2.3145948668628535) q[2];
ry(-1.072917003285331) q[4];
cx q[2],q[4];
ry(2.8300707941411876) q[2];
ry(-1.9837897366854778) q[4];
cx q[2],q[4];
ry(0.08310745406537556) q[4];
ry(-2.408092059038524) q[6];
cx q[4],q[6];
ry(-0.899609357402318) q[4];
ry(-1.3516099160161537) q[6];
cx q[4],q[6];
ry(1.7779007006925864) q[6];
ry(-0.5272029684746028) q[8];
cx q[6],q[8];
ry(-1.854587534870869) q[6];
ry(1.4230567913970216) q[8];
cx q[6],q[8];
ry(2.833237963541131) q[8];
ry(0.11140266276410049) q[10];
cx q[8],q[10];
ry(0.2684157310591946) q[8];
ry(-0.6843514064924746) q[10];
cx q[8],q[10];
ry(-1.0362413101216656) q[1];
ry(-1.463248182453114) q[3];
cx q[1],q[3];
ry(-0.4555635956746411) q[1];
ry(-1.6195499964896565) q[3];
cx q[1],q[3];
ry(2.0158719861216126) q[3];
ry(-0.5896825490262403) q[5];
cx q[3],q[5];
ry(-1.7843903338725653) q[3];
ry(-1.8218800944265219) q[5];
cx q[3],q[5];
ry(-1.6439245131567923) q[5];
ry(-1.9502088890397422) q[7];
cx q[5],q[7];
ry(-1.2167447573326946) q[5];
ry(-2.199610734710013) q[7];
cx q[5],q[7];
ry(-2.261472646101059) q[7];
ry(-1.0124505820692242) q[9];
cx q[7],q[9];
ry(2.401004021956157) q[7];
ry(-2.8339645593755773) q[9];
cx q[7],q[9];
ry(-3.075030515580883) q[9];
ry(-2.812950964365008) q[11];
cx q[9],q[11];
ry(-2.238952025597898) q[9];
ry(-2.7445795877746333) q[11];
cx q[9],q[11];
ry(2.4063197418383746) q[0];
ry(1.2277504322847868) q[3];
cx q[0],q[3];
ry(2.016607791925071) q[0];
ry(0.41421133853323777) q[3];
cx q[0],q[3];
ry(0.8700392315386258) q[1];
ry(1.6268539565916509) q[2];
cx q[1],q[2];
ry(-0.46340512151791896) q[1];
ry(1.9012039487891919) q[2];
cx q[1],q[2];
ry(0.9081269558238443) q[2];
ry(1.1954735396140315) q[5];
cx q[2],q[5];
ry(1.3900349964462757) q[2];
ry(0.42342044971432297) q[5];
cx q[2],q[5];
ry(2.8918569516338852) q[3];
ry(2.4709485238902684) q[4];
cx q[3],q[4];
ry(2.0572816730446335) q[3];
ry(-0.4303502981147448) q[4];
cx q[3],q[4];
ry(-2.515255880412441) q[4];
ry(1.2327968570637924) q[7];
cx q[4],q[7];
ry(-1.3943405546642822) q[4];
ry(1.193622281671094) q[7];
cx q[4],q[7];
ry(2.8081513878532505) q[5];
ry(0.2785369305948021) q[6];
cx q[5],q[6];
ry(0.7843149831584322) q[5];
ry(-2.0860269790664874) q[6];
cx q[5],q[6];
ry(1.913308857806837) q[6];
ry(-1.7232668902520665) q[9];
cx q[6],q[9];
ry(-0.2225587073053923) q[6];
ry(1.9412561284134942) q[9];
cx q[6],q[9];
ry(0.8585134154072502) q[7];
ry(0.22610044330747936) q[8];
cx q[7],q[8];
ry(2.7351977439222863) q[7];
ry(-0.3744457018037303) q[8];
cx q[7],q[8];
ry(2.8676733198830324) q[8];
ry(-2.418626770322002) q[11];
cx q[8],q[11];
ry(-0.3565186050038447) q[8];
ry(2.2295983635671934) q[11];
cx q[8],q[11];
ry(1.681459416050311) q[9];
ry(-2.599051215993624) q[10];
cx q[9],q[10];
ry(0.7634835092430706) q[9];
ry(-0.9447404877155589) q[10];
cx q[9],q[10];
ry(-2.318175208651752) q[0];
ry(0.42242581649488464) q[1];
cx q[0],q[1];
ry(1.388914767151566) q[0];
ry(-2.544958072852127) q[1];
cx q[0],q[1];
ry(-0.49880172224746394) q[2];
ry(-1.9612363180070478) q[3];
cx q[2],q[3];
ry(-2.780994808744548) q[2];
ry(2.6481093784097984) q[3];
cx q[2],q[3];
ry(-0.572182629617344) q[4];
ry(-2.5311762189719267) q[5];
cx q[4],q[5];
ry(2.8731322703850295) q[4];
ry(-2.098658727611742) q[5];
cx q[4],q[5];
ry(2.433515487698512) q[6];
ry(2.2382414144667777) q[7];
cx q[6],q[7];
ry(-2.5204666531171935) q[6];
ry(1.2033484578870401) q[7];
cx q[6],q[7];
ry(0.5185802942350968) q[8];
ry(-1.4334712823993867) q[9];
cx q[8],q[9];
ry(-2.2255764362503334) q[8];
ry(-2.6676248711507906) q[9];
cx q[8],q[9];
ry(1.1946711855229308) q[10];
ry(2.7730665651703985) q[11];
cx q[10],q[11];
ry(-1.4435782739738041) q[10];
ry(-0.7162066084299354) q[11];
cx q[10],q[11];
ry(1.5670896291213356) q[0];
ry(-0.6274136388806721) q[2];
cx q[0],q[2];
ry(2.9232413867492912) q[0];
ry(-1.9672688459950025) q[2];
cx q[0],q[2];
ry(0.15533707607180816) q[2];
ry(-0.47697135993958284) q[4];
cx q[2],q[4];
ry(1.0471528390410951) q[2];
ry(2.9784546636614935) q[4];
cx q[2],q[4];
ry(0.37487407626309555) q[4];
ry(-0.7525778884528793) q[6];
cx q[4],q[6];
ry(-2.903212093130148) q[4];
ry(0.7360690290769405) q[6];
cx q[4],q[6];
ry(-0.1423186544799595) q[6];
ry(2.6730269935934476) q[8];
cx q[6],q[8];
ry(-1.9660903159304137) q[6];
ry(-2.295310500038735) q[8];
cx q[6],q[8];
ry(3.1111215357767716) q[8];
ry(2.0821844231567628) q[10];
cx q[8],q[10];
ry(2.877653765146651) q[8];
ry(0.46662356534496746) q[10];
cx q[8],q[10];
ry(-2.0313450581112367) q[1];
ry(2.2353604615062155) q[3];
cx q[1],q[3];
ry(-1.2245855324088577) q[1];
ry(-0.8700414099116872) q[3];
cx q[1],q[3];
ry(0.9520847065444568) q[3];
ry(-2.8940979724851026) q[5];
cx q[3],q[5];
ry(0.8197243053140122) q[3];
ry(2.3295876078462663) q[5];
cx q[3],q[5];
ry(-0.07122942201605122) q[5];
ry(2.005179188248028) q[7];
cx q[5],q[7];
ry(0.8368053672056918) q[5];
ry(0.22511331179005015) q[7];
cx q[5],q[7];
ry(-2.231787950221948) q[7];
ry(-3.029898318923163) q[9];
cx q[7],q[9];
ry(2.63776417027732) q[7];
ry(2.9039914899383232) q[9];
cx q[7],q[9];
ry(-2.1862631220916624) q[9];
ry(3.0147320711744574) q[11];
cx q[9],q[11];
ry(-1.2475363237291788) q[9];
ry(0.8981369365051315) q[11];
cx q[9],q[11];
ry(-1.3638793212799003) q[0];
ry(2.8930562440171457) q[3];
cx q[0],q[3];
ry(0.24713209202280362) q[0];
ry(0.9938632810536471) q[3];
cx q[0],q[3];
ry(0.445334736889313) q[1];
ry(-1.2333730245119856) q[2];
cx q[1],q[2];
ry(-1.2471134749354373) q[1];
ry(-3.0136041958427113) q[2];
cx q[1],q[2];
ry(-2.5083873385585838) q[2];
ry(-3.12652294564789) q[5];
cx q[2],q[5];
ry(-0.5975492872882189) q[2];
ry(2.8335032575698227) q[5];
cx q[2],q[5];
ry(0.13530703484012166) q[3];
ry(-1.2086806613792078) q[4];
cx q[3],q[4];
ry(2.280159701664723) q[3];
ry(-2.4599066054251915) q[4];
cx q[3],q[4];
ry(-1.9396449203724124) q[4];
ry(0.3449379374883552) q[7];
cx q[4],q[7];
ry(2.7772169644905973) q[4];
ry(-1.7788204198231177) q[7];
cx q[4],q[7];
ry(-2.5812955556156223) q[5];
ry(1.4356633773880914) q[6];
cx q[5],q[6];
ry(-2.6326774269746984) q[5];
ry(-2.513071732485515) q[6];
cx q[5],q[6];
ry(-0.739337721837706) q[6];
ry(-0.1935757087827987) q[9];
cx q[6],q[9];
ry(-1.3347528410183498) q[6];
ry(-2.051036155324759) q[9];
cx q[6],q[9];
ry(2.9187016665626553) q[7];
ry(0.7654637953929521) q[8];
cx q[7],q[8];
ry(-2.5811606828714395) q[7];
ry(0.9982703280047627) q[8];
cx q[7],q[8];
ry(2.66388560961675) q[8];
ry(0.17819903211971067) q[11];
cx q[8],q[11];
ry(-1.2992572041488808) q[8];
ry(-2.456128045008432) q[11];
cx q[8],q[11];
ry(1.6575439328136108) q[9];
ry(1.6048166511625723) q[10];
cx q[9],q[10];
ry(1.2160456128621322) q[9];
ry(-1.0603295911769632) q[10];
cx q[9],q[10];
ry(-0.40595479603347656) q[0];
ry(-1.9309968716386994) q[1];
cx q[0],q[1];
ry(2.6481487330916913) q[0];
ry(-0.5329322087587272) q[1];
cx q[0],q[1];
ry(3.014663265175255) q[2];
ry(0.7057280175824577) q[3];
cx q[2],q[3];
ry(2.135255377485719) q[2];
ry(-1.9613625763544598) q[3];
cx q[2],q[3];
ry(-1.6673459013043823) q[4];
ry(-1.799479842644688) q[5];
cx q[4],q[5];
ry(1.998982830345411) q[4];
ry(-2.3791127071613727) q[5];
cx q[4],q[5];
ry(-0.1946447821168611) q[6];
ry(2.971525368214983) q[7];
cx q[6],q[7];
ry(-1.4185394341760382) q[6];
ry(0.6319176640176069) q[7];
cx q[6],q[7];
ry(-0.14620335733071776) q[8];
ry(1.526133481676068) q[9];
cx q[8],q[9];
ry(-2.755831658997345) q[8];
ry(2.5270636667946906) q[9];
cx q[8],q[9];
ry(-2.687196240091436) q[10];
ry(2.022284778983032) q[11];
cx q[10],q[11];
ry(-2.645195067925575) q[10];
ry(2.8894105691520986) q[11];
cx q[10],q[11];
ry(0.37422296175772907) q[0];
ry(0.46404239368011385) q[2];
cx q[0],q[2];
ry(2.7316700618120855) q[0];
ry(2.054766848102153) q[2];
cx q[0],q[2];
ry(-2.9792804771737864) q[2];
ry(1.3803570403184926) q[4];
cx q[2],q[4];
ry(0.29529093498157477) q[2];
ry(-2.146472302445021) q[4];
cx q[2],q[4];
ry(2.3999459543540267) q[4];
ry(0.8727585077429083) q[6];
cx q[4],q[6];
ry(-0.18855999300951107) q[4];
ry(-2.5872243572200024) q[6];
cx q[4],q[6];
ry(1.7884629269796963) q[6];
ry(-1.1925990166160094) q[8];
cx q[6],q[8];
ry(-0.4785924600373585) q[6];
ry(0.5713843971159944) q[8];
cx q[6],q[8];
ry(2.603646770425588) q[8];
ry(2.4235868892925936) q[10];
cx q[8],q[10];
ry(2.3656236866187195) q[8];
ry(-1.0208782580452596) q[10];
cx q[8],q[10];
ry(2.1449773503447136) q[1];
ry(-3.0714274374967907) q[3];
cx q[1],q[3];
ry(0.9551284539808842) q[1];
ry(-0.4241305798657007) q[3];
cx q[1],q[3];
ry(2.7183679461768113) q[3];
ry(-2.877295070446709) q[5];
cx q[3],q[5];
ry(-1.1803479577338705) q[3];
ry(0.48639045851306484) q[5];
cx q[3],q[5];
ry(1.1287054885709527) q[5];
ry(0.37793026360095094) q[7];
cx q[5],q[7];
ry(-0.43830137527844404) q[5];
ry(2.6042573591037987) q[7];
cx q[5],q[7];
ry(-0.9910701577431588) q[7];
ry(2.3063660246182676) q[9];
cx q[7],q[9];
ry(1.841740441116361) q[7];
ry(-1.9793998185773132) q[9];
cx q[7],q[9];
ry(-1.1400622890971706) q[9];
ry(-1.8161069130388414) q[11];
cx q[9],q[11];
ry(2.4191337991943467) q[9];
ry(-2.968115211050784) q[11];
cx q[9],q[11];
ry(-1.8676150966851317) q[0];
ry(-2.6396641637333955) q[3];
cx q[0],q[3];
ry(2.945946805326194) q[0];
ry(3.057559108025945) q[3];
cx q[0],q[3];
ry(2.069989252527133) q[1];
ry(2.0626083627446947) q[2];
cx q[1],q[2];
ry(2.6084750849401273) q[1];
ry(-2.6820790891813973) q[2];
cx q[1],q[2];
ry(-1.9216450498994584) q[2];
ry(0.6923800676027425) q[5];
cx q[2],q[5];
ry(1.3935250050476626) q[2];
ry(2.8769748969070554) q[5];
cx q[2],q[5];
ry(-2.0492071874185083) q[3];
ry(-0.7714950898051248) q[4];
cx q[3],q[4];
ry(-0.46426651967127625) q[3];
ry(0.9733611388430488) q[4];
cx q[3],q[4];
ry(1.874876480226514) q[4];
ry(1.177601361704961) q[7];
cx q[4],q[7];
ry(-1.467096498302464) q[4];
ry(-2.166968710192071) q[7];
cx q[4],q[7];
ry(-0.4850255306273654) q[5];
ry(0.5253947381281011) q[6];
cx q[5],q[6];
ry(-0.5007715957846086) q[5];
ry(1.9597836357632323) q[6];
cx q[5],q[6];
ry(-2.9435519801465486) q[6];
ry(-2.4707646272292294) q[9];
cx q[6],q[9];
ry(-2.0909458201656492) q[6];
ry(-1.19675145614245) q[9];
cx q[6],q[9];
ry(3.106800082348612) q[7];
ry(-1.5062672235987988) q[8];
cx q[7],q[8];
ry(2.804656481171548) q[7];
ry(-0.8648464976779673) q[8];
cx q[7],q[8];
ry(2.5004294630788277) q[8];
ry(2.944461661708299) q[11];
cx q[8],q[11];
ry(2.234633519117965) q[8];
ry(0.3920215163404633) q[11];
cx q[8],q[11];
ry(-3.1218949837583536) q[9];
ry(1.5244968288943808) q[10];
cx q[9],q[10];
ry(0.46654195266433446) q[9];
ry(-0.7072728728810755) q[10];
cx q[9],q[10];
ry(0.9436232158225426) q[0];
ry(2.562221609804967) q[1];
cx q[0],q[1];
ry(2.6187144830762703) q[0];
ry(-2.440435384058585) q[1];
cx q[0],q[1];
ry(2.499979347117202) q[2];
ry(0.29976562785438443) q[3];
cx q[2],q[3];
ry(-1.1845518437029345) q[2];
ry(2.80083312178384) q[3];
cx q[2],q[3];
ry(-2.927202409047203) q[4];
ry(-1.4688393644706126) q[5];
cx q[4],q[5];
ry(2.0875859844554467) q[4];
ry(0.8872616817433358) q[5];
cx q[4],q[5];
ry(0.40909665762382463) q[6];
ry(0.5352984489576798) q[7];
cx q[6],q[7];
ry(-2.491876636738178) q[6];
ry(-1.3009878016474532) q[7];
cx q[6],q[7];
ry(1.9616174777857012) q[8];
ry(1.655342838251272) q[9];
cx q[8],q[9];
ry(-2.1985194422026315) q[8];
ry(-2.457944906960035) q[9];
cx q[8],q[9];
ry(-2.8582550702261402) q[10];
ry(2.47526392465713) q[11];
cx q[10],q[11];
ry(-0.5590920974002046) q[10];
ry(-1.0174153643576185) q[11];
cx q[10],q[11];
ry(1.8330649021000598) q[0];
ry(0.764679147278231) q[2];
cx q[0],q[2];
ry(-0.5926947187680928) q[0];
ry(2.065071958392606) q[2];
cx q[0],q[2];
ry(3.1341816046815087) q[2];
ry(1.9851259215197112) q[4];
cx q[2],q[4];
ry(-0.3660055173728088) q[2];
ry(-0.7433244066418663) q[4];
cx q[2],q[4];
ry(-0.3497080627086025) q[4];
ry(-0.9113698319820163) q[6];
cx q[4],q[6];
ry(-1.3320236270406625) q[4];
ry(0.18459160467523003) q[6];
cx q[4],q[6];
ry(-1.5717216323766716) q[6];
ry(0.5976648455428969) q[8];
cx q[6],q[8];
ry(-2.204560353540729) q[6];
ry(1.6468626241199669) q[8];
cx q[6],q[8];
ry(0.249696893568979) q[8];
ry(-2.839159549629956) q[10];
cx q[8],q[10];
ry(1.7030690198596752) q[8];
ry(-2.168871062229801) q[10];
cx q[8],q[10];
ry(0.9997288764753405) q[1];
ry(2.580193239658524) q[3];
cx q[1],q[3];
ry(-1.907203388669608) q[1];
ry(1.8390719278367036) q[3];
cx q[1],q[3];
ry(-3.0305876288224893) q[3];
ry(1.9374398518741547) q[5];
cx q[3],q[5];
ry(1.5889851991062907) q[3];
ry(0.8936545027879473) q[5];
cx q[3],q[5];
ry(-0.437972581093387) q[5];
ry(2.054851910991162) q[7];
cx q[5],q[7];
ry(-2.0287933917367504) q[5];
ry(-1.7789607456451269) q[7];
cx q[5],q[7];
ry(-2.9068007122428803) q[7];
ry(0.6156290608643262) q[9];
cx q[7],q[9];
ry(-2.2953938449132827) q[7];
ry(0.8218560978758891) q[9];
cx q[7],q[9];
ry(-0.11220467953002627) q[9];
ry(2.3141842681417404) q[11];
cx q[9],q[11];
ry(-1.390703935714929) q[9];
ry(2.0766909167165952) q[11];
cx q[9],q[11];
ry(1.4729918674793494) q[0];
ry(0.9332796369030723) q[3];
cx q[0],q[3];
ry(2.764857010394937) q[0];
ry(1.7634496263761097) q[3];
cx q[0],q[3];
ry(0.7964140665983706) q[1];
ry(-2.823610054791037) q[2];
cx q[1],q[2];
ry(2.6913167322917264) q[1];
ry(-1.9892474601854389) q[2];
cx q[1],q[2];
ry(2.829756191086327) q[2];
ry(-0.9143421234803077) q[5];
cx q[2],q[5];
ry(0.5960955416504052) q[2];
ry(-1.9393075741504582) q[5];
cx q[2],q[5];
ry(-0.733387986811695) q[3];
ry(-1.6774956168059418) q[4];
cx q[3],q[4];
ry(2.51404383354229) q[3];
ry(0.08714696443287767) q[4];
cx q[3],q[4];
ry(1.4334354673030978) q[4];
ry(-2.7774124863914413) q[7];
cx q[4],q[7];
ry(-2.842269204316974) q[4];
ry(-2.045958271415137) q[7];
cx q[4],q[7];
ry(-1.979461904250912) q[5];
ry(1.9964138306458292) q[6];
cx q[5],q[6];
ry(-0.20975548587290124) q[5];
ry(-1.4181615084251735) q[6];
cx q[5],q[6];
ry(0.013642991820741024) q[6];
ry(0.7103561311700981) q[9];
cx q[6],q[9];
ry(-2.2848861071954087) q[6];
ry(-2.38660183722489) q[9];
cx q[6],q[9];
ry(1.2894945190358795) q[7];
ry(-1.578281780840613) q[8];
cx q[7],q[8];
ry(0.595888061786498) q[7];
ry(2.4691581643782965) q[8];
cx q[7],q[8];
ry(-2.4508006874957964) q[8];
ry(-1.9933099988136689) q[11];
cx q[8],q[11];
ry(-2.2141259827016517) q[8];
ry(2.68822975041562) q[11];
cx q[8],q[11];
ry(2.502016375289901) q[9];
ry(-0.49860087884897936) q[10];
cx q[9],q[10];
ry(-1.8227855642028166) q[9];
ry(1.4485421694755791) q[10];
cx q[9],q[10];
ry(0.3644321730573557) q[0];
ry(-1.5701649165440774) q[1];
cx q[0],q[1];
ry(1.9494548779280985) q[0];
ry(-1.154594001052029) q[1];
cx q[0],q[1];
ry(0.41032576086574846) q[2];
ry(-0.9092064292990321) q[3];
cx q[2],q[3];
ry(1.7851665643251238) q[2];
ry(2.3252819117402095) q[3];
cx q[2],q[3];
ry(-0.8947289952336712) q[4];
ry(-2.4979391530203268) q[5];
cx q[4],q[5];
ry(-2.9273874356062715) q[4];
ry(-2.879438646451317) q[5];
cx q[4],q[5];
ry(-2.6722750147797525) q[6];
ry(-2.8270032613916847) q[7];
cx q[6],q[7];
ry(1.055100190476388) q[6];
ry(-1.3672936306130659) q[7];
cx q[6],q[7];
ry(2.593220073979573) q[8];
ry(0.5228598692463571) q[9];
cx q[8],q[9];
ry(1.978302458376585) q[8];
ry(-0.11760712788512853) q[9];
cx q[8],q[9];
ry(0.8813780701928845) q[10];
ry(0.588884202982749) q[11];
cx q[10],q[11];
ry(-0.20717237447258516) q[10];
ry(-1.6848353613160358) q[11];
cx q[10],q[11];
ry(0.7626701588008524) q[0];
ry(0.24638361235858766) q[2];
cx q[0],q[2];
ry(-2.2947158995494905) q[0];
ry(1.6507378617154114) q[2];
cx q[0],q[2];
ry(0.5696700000083945) q[2];
ry(2.9462190197129807) q[4];
cx q[2],q[4];
ry(1.892878734214804) q[2];
ry(2.383595406645196) q[4];
cx q[2],q[4];
ry(-2.138350347357478) q[4];
ry(0.654945232702552) q[6];
cx q[4],q[6];
ry(-0.4456329668071941) q[4];
ry(-2.487279577810487) q[6];
cx q[4],q[6];
ry(-0.5987622693868975) q[6];
ry(2.9883134848422572) q[8];
cx q[6],q[8];
ry(2.3294502208044103) q[6];
ry(2.0362285863199974) q[8];
cx q[6],q[8];
ry(-1.1154036925489246) q[8];
ry(1.5485507293583771) q[10];
cx q[8],q[10];
ry(1.9969131554122692) q[8];
ry(2.6436369081294537) q[10];
cx q[8],q[10];
ry(-2.2513088086941497) q[1];
ry(2.559630875498364) q[3];
cx q[1],q[3];
ry(-1.1442599288226654) q[1];
ry(-0.3789861805906751) q[3];
cx q[1],q[3];
ry(-0.06493446340544633) q[3];
ry(-0.18437384095848766) q[5];
cx q[3],q[5];
ry(-1.9557759303227584) q[3];
ry(0.17229068690924654) q[5];
cx q[3],q[5];
ry(-0.8830179600039116) q[5];
ry(-2.7573253236759037) q[7];
cx q[5],q[7];
ry(-2.787835206899636) q[5];
ry(-1.4635892697523465) q[7];
cx q[5],q[7];
ry(2.194425664594057) q[7];
ry(0.6894046941073267) q[9];
cx q[7],q[9];
ry(0.7310886775894055) q[7];
ry(-1.1335903040915003) q[9];
cx q[7],q[9];
ry(0.8033546359927188) q[9];
ry(-2.4586495165778985) q[11];
cx q[9],q[11];
ry(1.5484982603653072) q[9];
ry(0.8686355443092137) q[11];
cx q[9],q[11];
ry(2.5211663164492415) q[0];
ry(1.3792318106670463) q[3];
cx q[0],q[3];
ry(-2.4561532733933147) q[0];
ry(-0.45297359996974307) q[3];
cx q[0],q[3];
ry(-2.9089517388885313) q[1];
ry(-0.07756502316899602) q[2];
cx q[1],q[2];
ry(2.2931655915761033) q[1];
ry(-0.9774450005185902) q[2];
cx q[1],q[2];
ry(-0.8321823766993149) q[2];
ry(2.9888492043087687) q[5];
cx q[2],q[5];
ry(-0.22913840895654608) q[2];
ry(3.076811068099995) q[5];
cx q[2],q[5];
ry(1.8143665824878246) q[3];
ry(1.1004530349954547) q[4];
cx q[3],q[4];
ry(2.2609891196386207) q[3];
ry(-1.8965963770276062) q[4];
cx q[3],q[4];
ry(-1.999078067169484) q[4];
ry(-1.6782055651430827) q[7];
cx q[4],q[7];
ry(-2.1720959566757436) q[4];
ry(-1.3385187551264914) q[7];
cx q[4],q[7];
ry(-2.9647305390104037) q[5];
ry(-3.0638600510924534) q[6];
cx q[5],q[6];
ry(-2.609918543260031) q[5];
ry(-0.5351814663305765) q[6];
cx q[5],q[6];
ry(-2.44287790054281) q[6];
ry(2.728312029059517) q[9];
cx q[6],q[9];
ry(0.21879074202879784) q[6];
ry(-3.023848513934561) q[9];
cx q[6],q[9];
ry(1.2182946977124203) q[7];
ry(-0.5853752955452238) q[8];
cx q[7],q[8];
ry(-2.255250477437361) q[7];
ry(2.9078432442295026) q[8];
cx q[7],q[8];
ry(-0.4602974354119249) q[8];
ry(1.4859414771718422) q[11];
cx q[8],q[11];
ry(-2.755989551098481) q[8];
ry(1.6805120910150984) q[11];
cx q[8],q[11];
ry(-0.7456708395286902) q[9];
ry(2.213781347477319) q[10];
cx q[9],q[10];
ry(3.023329802631666) q[9];
ry(-0.6605235237999036) q[10];
cx q[9],q[10];
ry(-1.0731701246870864) q[0];
ry(3.0557350746670386) q[1];
cx q[0],q[1];
ry(2.988464223611823) q[0];
ry(-2.5816230840623273) q[1];
cx q[0],q[1];
ry(-2.2340706986253784) q[2];
ry(-2.852442941250237) q[3];
cx q[2],q[3];
ry(2.754462684452681) q[2];
ry(2.8142793775142585) q[3];
cx q[2],q[3];
ry(0.5626817634897925) q[4];
ry(-0.41046108098251993) q[5];
cx q[4],q[5];
ry(-0.48624031297020126) q[4];
ry(-2.4640459973488418) q[5];
cx q[4],q[5];
ry(-2.122490680769922) q[6];
ry(-2.0147344662109066) q[7];
cx q[6],q[7];
ry(1.8955090998200452) q[6];
ry(1.8781107812452704) q[7];
cx q[6],q[7];
ry(-2.1624324038253224) q[8];
ry(-2.689639178908186) q[9];
cx q[8],q[9];
ry(1.8311103546739795) q[8];
ry(0.19006268671242954) q[9];
cx q[8],q[9];
ry(-3.054601265929342) q[10];
ry(-2.7849411746457498) q[11];
cx q[10],q[11];
ry(1.4118853966352547) q[10];
ry(-2.1568528893609726) q[11];
cx q[10],q[11];
ry(2.0354027533112875) q[0];
ry(-1.8428847797708254) q[2];
cx q[0],q[2];
ry(-0.2363791203337433) q[0];
ry(-0.7848720355331417) q[2];
cx q[0],q[2];
ry(0.6340017406350587) q[2];
ry(-0.6535315684257907) q[4];
cx q[2],q[4];
ry(1.926272348536119) q[2];
ry(0.7788653036030909) q[4];
cx q[2],q[4];
ry(-0.6406276523345737) q[4];
ry(-2.3237688309646836) q[6];
cx q[4],q[6];
ry(-0.3531873755248265) q[4];
ry(-1.033077188518226) q[6];
cx q[4],q[6];
ry(-2.8525142447263083) q[6];
ry(2.962948352477645) q[8];
cx q[6],q[8];
ry(-1.084744576814429) q[6];
ry(-0.4737336746972858) q[8];
cx q[6],q[8];
ry(1.1027177961703607) q[8];
ry(-2.677396732508255) q[10];
cx q[8],q[10];
ry(3.010106167098597) q[8];
ry(-2.555092386936147) q[10];
cx q[8],q[10];
ry(-3.128431962429101) q[1];
ry(-2.9875137245989967) q[3];
cx q[1],q[3];
ry(2.999906004720424) q[1];
ry(2.5283922828075323) q[3];
cx q[1],q[3];
ry(-0.10973290580289863) q[3];
ry(2.90934545803999) q[5];
cx q[3],q[5];
ry(2.1105117979611014) q[3];
ry(-0.12622044266242405) q[5];
cx q[3],q[5];
ry(2.7358047997076094) q[5];
ry(-1.3335052894521224) q[7];
cx q[5],q[7];
ry(-2.0409574530288346) q[5];
ry(-0.7119404921933699) q[7];
cx q[5],q[7];
ry(1.7722775838144198) q[7];
ry(2.3696796194045278) q[9];
cx q[7],q[9];
ry(-1.3371714893194286) q[7];
ry(-1.9694894721361549) q[9];
cx q[7],q[9];
ry(-2.9843381505621798) q[9];
ry(0.4032190908359988) q[11];
cx q[9],q[11];
ry(0.6329772243235556) q[9];
ry(0.8328741292242425) q[11];
cx q[9],q[11];
ry(-1.5599535154499151) q[0];
ry(1.9823527943747665) q[3];
cx q[0],q[3];
ry(-1.6536804208222087) q[0];
ry(-2.971228517474047) q[3];
cx q[0],q[3];
ry(2.422200721256646) q[1];
ry(-1.492034583998644) q[2];
cx q[1],q[2];
ry(2.2187513631721303) q[1];
ry(0.8293733948450788) q[2];
cx q[1],q[2];
ry(-1.2775899825056491) q[2];
ry(0.41121001073660457) q[5];
cx q[2],q[5];
ry(2.0741871444857938) q[2];
ry(-2.88007067417383) q[5];
cx q[2],q[5];
ry(-1.7886806949984289) q[3];
ry(-2.8527214470047517) q[4];
cx q[3],q[4];
ry(-1.2521352990510524) q[3];
ry(-0.6976179900106336) q[4];
cx q[3],q[4];
ry(1.8811375753139186) q[4];
ry(2.862993113463472) q[7];
cx q[4],q[7];
ry(-1.2289630781574548) q[4];
ry(-0.1745341549957955) q[7];
cx q[4],q[7];
ry(-1.166133818757008) q[5];
ry(1.716170223439069) q[6];
cx q[5],q[6];
ry(-2.8075415564737747) q[5];
ry(-2.433171910290056) q[6];
cx q[5],q[6];
ry(-1.2387998618827378) q[6];
ry(-1.431024177842368) q[9];
cx q[6],q[9];
ry(-0.17647591453013778) q[6];
ry(-0.30767547897078007) q[9];
cx q[6],q[9];
ry(-0.13296495539098263) q[7];
ry(-1.6709798674348075) q[8];
cx q[7],q[8];
ry(-2.1755045477391968) q[7];
ry(3.1023838501447494) q[8];
cx q[7],q[8];
ry(-0.6888041282348851) q[8];
ry(-1.0576997242631223) q[11];
cx q[8],q[11];
ry(1.3672755166725548) q[8];
ry(2.0323873369367496) q[11];
cx q[8],q[11];
ry(2.1673033972139617) q[9];
ry(-2.830145040257223) q[10];
cx q[9],q[10];
ry(-3.0923746889626145) q[9];
ry(1.1966701561215456) q[10];
cx q[9],q[10];
ry(2.058137122224619) q[0];
ry(-0.22015496664407944) q[1];
cx q[0],q[1];
ry(1.651867508274647) q[0];
ry(-0.8084060942329607) q[1];
cx q[0],q[1];
ry(1.6878941445932272) q[2];
ry(-0.41929396262954466) q[3];
cx q[2],q[3];
ry(-2.17567108231973) q[2];
ry(-0.25539723673105286) q[3];
cx q[2],q[3];
ry(3.09473760858376) q[4];
ry(-1.1323551139987997) q[5];
cx q[4],q[5];
ry(-2.2949818483394124) q[4];
ry(-2.177908912725821) q[5];
cx q[4],q[5];
ry(2.01280775629769) q[6];
ry(1.676161621168057) q[7];
cx q[6],q[7];
ry(-1.3864002127785824) q[6];
ry(-0.8773182572326162) q[7];
cx q[6],q[7];
ry(1.4652678218587334) q[8];
ry(-2.59290855712942) q[9];
cx q[8],q[9];
ry(-2.7105985258547896) q[8];
ry(-0.9265315500104538) q[9];
cx q[8],q[9];
ry(-1.9678322582948649) q[10];
ry(-2.5225584637222362) q[11];
cx q[10],q[11];
ry(2.2696589610179014) q[10];
ry(-1.6072860214641236) q[11];
cx q[10],q[11];
ry(1.5729564902995943) q[0];
ry(-0.1132660627327784) q[2];
cx q[0],q[2];
ry(-0.6529010809154082) q[0];
ry(-0.5020515090166341) q[2];
cx q[0],q[2];
ry(-0.3133933700611907) q[2];
ry(-1.865530167181739) q[4];
cx q[2],q[4];
ry(-2.05329526773533) q[2];
ry(1.8268437623497684) q[4];
cx q[2],q[4];
ry(-1.413101710296755) q[4];
ry(-2.936972189768462) q[6];
cx q[4],q[6];
ry(-0.7650920311578034) q[4];
ry(-0.8444805871390296) q[6];
cx q[4],q[6];
ry(1.4811228214695284) q[6];
ry(-1.8287668334146234) q[8];
cx q[6],q[8];
ry(2.442633702519038) q[6];
ry(1.2243183452922395) q[8];
cx q[6],q[8];
ry(-1.2682945232948921) q[8];
ry(-2.0399526123832796) q[10];
cx q[8],q[10];
ry(3.0460399198207426) q[8];
ry(2.6494280808221222) q[10];
cx q[8],q[10];
ry(-1.1702883060867246) q[1];
ry(2.185450194396863) q[3];
cx q[1],q[3];
ry(2.2951695761283135) q[1];
ry(-0.502873842510526) q[3];
cx q[1],q[3];
ry(0.6339536397837442) q[3];
ry(-0.14385187931290844) q[5];
cx q[3],q[5];
ry(-0.5800028702868243) q[3];
ry(1.6997590668330766) q[5];
cx q[3],q[5];
ry(-0.28267067452592015) q[5];
ry(-2.6949524000844383) q[7];
cx q[5],q[7];
ry(1.9528236835596438) q[5];
ry(1.8917000412071043) q[7];
cx q[5],q[7];
ry(0.9445343801899037) q[7];
ry(-2.194881499928027) q[9];
cx q[7],q[9];
ry(2.5722799236945275) q[7];
ry(1.3377657340840128) q[9];
cx q[7],q[9];
ry(0.3255962690653495) q[9];
ry(-0.15380674580018966) q[11];
cx q[9],q[11];
ry(-2.253323651538038) q[9];
ry(-2.6125513625552923) q[11];
cx q[9],q[11];
ry(1.665384255762977) q[0];
ry(-1.1216623742169651) q[3];
cx q[0],q[3];
ry(-1.6071563008076786) q[0];
ry(-1.5703669088783485) q[3];
cx q[0],q[3];
ry(1.0599794940493026) q[1];
ry(-0.00689050393940871) q[2];
cx q[1],q[2];
ry(-1.8318915367010185) q[1];
ry(-1.3127255383437575) q[2];
cx q[1],q[2];
ry(0.23781651564105655) q[2];
ry(2.9169082274197193) q[5];
cx q[2],q[5];
ry(0.5677420968371993) q[2];
ry(2.678657664202516) q[5];
cx q[2],q[5];
ry(-1.444670065247494) q[3];
ry(0.2764417088268738) q[4];
cx q[3],q[4];
ry(-0.9884762046095332) q[3];
ry(2.608921166070929) q[4];
cx q[3],q[4];
ry(-0.2826625910981342) q[4];
ry(-1.157995162949768) q[7];
cx q[4],q[7];
ry(0.3905993568163347) q[4];
ry(-0.6529948381601471) q[7];
cx q[4],q[7];
ry(-1.6406464545300707) q[5];
ry(-0.21319276583732685) q[6];
cx q[5],q[6];
ry(0.7735493615199341) q[5];
ry(0.7033446563328285) q[6];
cx q[5],q[6];
ry(-0.5084674310897377) q[6];
ry(3.0272844161108843) q[9];
cx q[6],q[9];
ry(1.2254435773724461) q[6];
ry(-0.5541586169601631) q[9];
cx q[6],q[9];
ry(0.9176227573900523) q[7];
ry(1.6186350837973054) q[8];
cx q[7],q[8];
ry(-0.16659711455483234) q[7];
ry(3.068868005399062) q[8];
cx q[7],q[8];
ry(-1.9612185931090438) q[8];
ry(0.1365342103133343) q[11];
cx q[8],q[11];
ry(1.2408336774851796) q[8];
ry(1.4117416629447677) q[11];
cx q[8],q[11];
ry(1.5005170638805971) q[9];
ry(1.8064378093358089) q[10];
cx q[9],q[10];
ry(-0.19970771732122294) q[9];
ry(0.2778602054608701) q[10];
cx q[9],q[10];
ry(0.5091776383154971) q[0];
ry(0.4896996078786752) q[1];
cx q[0],q[1];
ry(-1.9800932053441338) q[0];
ry(-1.75099173408326) q[1];
cx q[0],q[1];
ry(-2.868064245364961) q[2];
ry(0.5427481742485565) q[3];
cx q[2],q[3];
ry(-2.30145792237241) q[2];
ry(1.5220466966526986) q[3];
cx q[2],q[3];
ry(0.8915231280016734) q[4];
ry(0.9020251929974409) q[5];
cx q[4],q[5];
ry(-1.9136126369356117) q[4];
ry(-1.1581459681316206) q[5];
cx q[4],q[5];
ry(1.4189968984276682) q[6];
ry(-1.6340613741541885) q[7];
cx q[6],q[7];
ry(0.10615787289609191) q[6];
ry(-1.7102297234235488) q[7];
cx q[6],q[7];
ry(-1.309802304212603) q[8];
ry(-0.43580610504395334) q[9];
cx q[8],q[9];
ry(-2.011373992135261) q[8];
ry(0.8941135541675801) q[9];
cx q[8],q[9];
ry(0.19710457873990883) q[10];
ry(0.9634696915752302) q[11];
cx q[10],q[11];
ry(-2.4142263232504515) q[10];
ry(2.1084320373521273) q[11];
cx q[10],q[11];
ry(3.1237676131298464) q[0];
ry(-1.6829964515188234) q[2];
cx q[0],q[2];
ry(2.8996612703082705) q[0];
ry(-2.8472175289607065) q[2];
cx q[0],q[2];
ry(1.011510951881803) q[2];
ry(2.2879043132555585) q[4];
cx q[2],q[4];
ry(1.7305904517696886) q[2];
ry(-1.3104540978654973) q[4];
cx q[2],q[4];
ry(2.503207665219711) q[4];
ry(-0.48212080865647583) q[6];
cx q[4],q[6];
ry(-0.4536027677355673) q[4];
ry(2.1870287669186945) q[6];
cx q[4],q[6];
ry(-2.0742489587604283) q[6];
ry(2.7979870996773677) q[8];
cx q[6],q[8];
ry(0.7895010230338934) q[6];
ry(1.3346807792525457) q[8];
cx q[6],q[8];
ry(-0.10647617810144622) q[8];
ry(-0.9893029272746853) q[10];
cx q[8],q[10];
ry(1.6330980294529334) q[8];
ry(-1.8695106407978397) q[10];
cx q[8],q[10];
ry(0.25509645834319356) q[1];
ry(0.388787467857363) q[3];
cx q[1],q[3];
ry(-1.4043361720933951) q[1];
ry(-2.5368637405609533) q[3];
cx q[1],q[3];
ry(0.07333529851664979) q[3];
ry(2.2996029015429706) q[5];
cx q[3],q[5];
ry(-2.002295703501291) q[3];
ry(0.5398448870597959) q[5];
cx q[3],q[5];
ry(-2.7399113212787287) q[5];
ry(-0.7512406674041011) q[7];
cx q[5],q[7];
ry(0.4074622450730709) q[5];
ry(2.0757486180382676) q[7];
cx q[5],q[7];
ry(2.44769146050782) q[7];
ry(2.9564088117093585) q[9];
cx q[7],q[9];
ry(2.6978551658264416) q[7];
ry(0.24243601427086148) q[9];
cx q[7],q[9];
ry(1.7196169184829766) q[9];
ry(0.3377020088401145) q[11];
cx q[9],q[11];
ry(-1.0661059277492975) q[9];
ry(-3.024280618231517) q[11];
cx q[9],q[11];
ry(1.2704970650407903) q[0];
ry(-2.2994772013950313) q[3];
cx q[0],q[3];
ry(-0.9704058057483694) q[0];
ry(-1.4826405931662405) q[3];
cx q[0],q[3];
ry(-0.07678142070198478) q[1];
ry(-0.8507882216868983) q[2];
cx q[1],q[2];
ry(-0.5481311815355276) q[1];
ry(1.4869248662977173) q[2];
cx q[1],q[2];
ry(1.3121071079206323) q[2];
ry(1.1529840992409106) q[5];
cx q[2],q[5];
ry(0.6353459663396226) q[2];
ry(-0.5566528164954878) q[5];
cx q[2],q[5];
ry(-0.5736590581099201) q[3];
ry(0.9125783707788644) q[4];
cx q[3],q[4];
ry(-1.012443092608729) q[3];
ry(-0.7819257139019055) q[4];
cx q[3],q[4];
ry(-2.4273483426885276) q[4];
ry(0.7342805162825339) q[7];
cx q[4],q[7];
ry(-2.0235703037173622) q[4];
ry(2.1053985036264917) q[7];
cx q[4],q[7];
ry(-2.5896744862674033) q[5];
ry(1.0627507249544328) q[6];
cx q[5],q[6];
ry(-2.8829953600055904) q[5];
ry(1.6036060610131067) q[6];
cx q[5],q[6];
ry(-1.6185741420276687) q[6];
ry(-3.0212550626141192) q[9];
cx q[6],q[9];
ry(1.4886758299359544) q[6];
ry(2.6779744196175543) q[9];
cx q[6],q[9];
ry(-2.38891260982739) q[7];
ry(-0.09954135558415178) q[8];
cx q[7],q[8];
ry(2.1470697942160597) q[7];
ry(0.5548213767878796) q[8];
cx q[7],q[8];
ry(2.7480857748368086) q[8];
ry(1.8993862948870748) q[11];
cx q[8],q[11];
ry(2.0777649434526166) q[8];
ry(1.3443045736438135) q[11];
cx q[8],q[11];
ry(2.2803746699482987) q[9];
ry(0.5980809008646988) q[10];
cx q[9],q[10];
ry(-1.0243841995499547) q[9];
ry(-0.7253450122825456) q[10];
cx q[9],q[10];
ry(-0.3966408018746315) q[0];
ry(1.8991782279037315) q[1];
cx q[0],q[1];
ry(2.2254478359673566) q[0];
ry(2.8844422713325995) q[1];
cx q[0],q[1];
ry(0.5942534962883211) q[2];
ry(2.8955590753499396) q[3];
cx q[2],q[3];
ry(0.27235962167783256) q[2];
ry(-1.3945049020997582) q[3];
cx q[2],q[3];
ry(1.8797226266398306) q[4];
ry(-1.8295463605215623) q[5];
cx q[4],q[5];
ry(1.0731145074333517) q[4];
ry(-1.341118781265835) q[5];
cx q[4],q[5];
ry(-0.571669151358627) q[6];
ry(0.918757598910913) q[7];
cx q[6],q[7];
ry(1.767416962282537) q[6];
ry(-1.0177839819318948) q[7];
cx q[6],q[7];
ry(-2.9528435916253253) q[8];
ry(1.646027883009527) q[9];
cx q[8],q[9];
ry(0.16437591140441654) q[8];
ry(-0.4967906890590479) q[9];
cx q[8],q[9];
ry(1.9409020213629429) q[10];
ry(-1.9826381251765948) q[11];
cx q[10],q[11];
ry(1.4256270397735387) q[10];
ry(1.663999408718257) q[11];
cx q[10],q[11];
ry(-1.408333581132923) q[0];
ry(-2.733307825180565) q[2];
cx q[0],q[2];
ry(-0.9280237588496636) q[0];
ry(1.3116968070222006) q[2];
cx q[0],q[2];
ry(-2.604584329340513) q[2];
ry(1.7491122886732209) q[4];
cx q[2],q[4];
ry(0.3363212793896352) q[2];
ry(2.573980403811431) q[4];
cx q[2],q[4];
ry(2.423772376732801) q[4];
ry(-2.0036940282078404) q[6];
cx q[4],q[6];
ry(0.6135566459278319) q[4];
ry(0.44988359657344734) q[6];
cx q[4],q[6];
ry(-0.7872546707209578) q[6];
ry(-0.04635743800632586) q[8];
cx q[6],q[8];
ry(-2.7996830718338996) q[6];
ry(2.666972999803982) q[8];
cx q[6],q[8];
ry(1.8655157283333073) q[8];
ry(2.579448258861285) q[10];
cx q[8],q[10];
ry(-2.117430812190112) q[8];
ry(-2.4616628326178907) q[10];
cx q[8],q[10];
ry(-0.3286355779736486) q[1];
ry(1.1836454161858452) q[3];
cx q[1],q[3];
ry(-0.927253936164425) q[1];
ry(2.9962664623779696) q[3];
cx q[1],q[3];
ry(1.1947663675276992) q[3];
ry(-1.3456602482501958) q[5];
cx q[3],q[5];
ry(1.3579757463673907) q[3];
ry(-1.08277212371568) q[5];
cx q[3],q[5];
ry(-0.17652209064663182) q[5];
ry(1.6029585227430634) q[7];
cx q[5],q[7];
ry(0.8426008914503766) q[5];
ry(-1.6322779020459215) q[7];
cx q[5],q[7];
ry(1.0917014721424565) q[7];
ry(-0.7538987376404461) q[9];
cx q[7],q[9];
ry(-1.1861476550350547) q[7];
ry(-1.8961284087301964) q[9];
cx q[7],q[9];
ry(-2.211422958048031) q[9];
ry(-1.3749804988929426) q[11];
cx q[9],q[11];
ry(-2.3733595089489143) q[9];
ry(-3.028222032412599) q[11];
cx q[9],q[11];
ry(2.764798552770962) q[0];
ry(1.573226232966727) q[3];
cx q[0],q[3];
ry(2.5402388256257495) q[0];
ry(2.738491515595993) q[3];
cx q[0],q[3];
ry(-2.6172634947842623) q[1];
ry(-0.19382207814283098) q[2];
cx q[1],q[2];
ry(-0.37026560585910673) q[1];
ry(-2.7923793035570967) q[2];
cx q[1],q[2];
ry(2.1971212044103345) q[2];
ry(2.445649942324829) q[5];
cx q[2],q[5];
ry(-2.9885329999208334) q[2];
ry(-1.8702081040868337) q[5];
cx q[2],q[5];
ry(3.053199693744583) q[3];
ry(-2.9235734527787707) q[4];
cx q[3],q[4];
ry(-2.592993250744465) q[3];
ry(-2.5946284573714453) q[4];
cx q[3],q[4];
ry(1.5953116546656927) q[4];
ry(-0.4503974648420283) q[7];
cx q[4],q[7];
ry(2.6521617050911135) q[4];
ry(1.1309948654512132) q[7];
cx q[4],q[7];
ry(-0.5386889442571415) q[5];
ry(-1.6680937179195026) q[6];
cx q[5],q[6];
ry(-2.8642054484908086) q[5];
ry(-0.8159463085904748) q[6];
cx q[5],q[6];
ry(-0.7740824752443274) q[6];
ry(-1.5352968592021323) q[9];
cx q[6],q[9];
ry(0.28429183886991094) q[6];
ry(-0.6567465961911804) q[9];
cx q[6],q[9];
ry(2.9970885196531474) q[7];
ry(2.711326526846315) q[8];
cx q[7],q[8];
ry(-2.9287493470806085) q[7];
ry(-2.1764148860983443) q[8];
cx q[7],q[8];
ry(-3.0194767160301303) q[8];
ry(0.1687947983822795) q[11];
cx q[8],q[11];
ry(-0.5500202753466517) q[8];
ry(1.6260949383720087) q[11];
cx q[8],q[11];
ry(-2.8787310570251035) q[9];
ry(-0.2267983168624017) q[10];
cx q[9],q[10];
ry(2.5246300355509974) q[9];
ry(1.650153675407104) q[10];
cx q[9],q[10];
ry(-1.3951335980293056) q[0];
ry(1.0976011402070798) q[1];
ry(0.466365092180082) q[2];
ry(2.2936911855880724) q[3];
ry(-1.8960002830261267) q[4];
ry(2.9702901601552636) q[5];
ry(2.1213402279365674) q[6];
ry(1.1826608827035072) q[7];
ry(-0.8157486712818273) q[8];
ry(2.4525575557168127) q[9];
ry(-0.22213822715610632) q[10];
ry(-3.0002899182715934) q[11];