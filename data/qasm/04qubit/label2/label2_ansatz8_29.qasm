OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.14049543996174482) q[0];
ry(2.285586685538431) q[1];
cx q[0],q[1];
ry(2.6978898873592763) q[0];
ry(-2.8107167898830223) q[1];
cx q[0],q[1];
ry(2.496340618466118) q[2];
ry(0.17963713221272748) q[3];
cx q[2],q[3];
ry(-1.6062549902181336) q[2];
ry(1.5625485503540442) q[3];
cx q[2],q[3];
ry(0.07039463165801685) q[0];
ry(-0.3359464563417873) q[2];
cx q[0],q[2];
ry(2.369950441376813) q[0];
ry(1.8348471392007357) q[2];
cx q[0],q[2];
ry(1.5312605103337509) q[1];
ry(-0.34319957680440005) q[3];
cx q[1],q[3];
ry(0.3037036781331599) q[1];
ry(0.9648283192426711) q[3];
cx q[1],q[3];
ry(-1.4755592157154567) q[0];
ry(-0.6900833988449341) q[1];
cx q[0],q[1];
ry(-0.7397177804388813) q[0];
ry(-1.0700534352513555) q[1];
cx q[0],q[1];
ry(-0.6267654307594798) q[2];
ry(2.963829967076248) q[3];
cx q[2],q[3];
ry(-2.8303707736622052) q[2];
ry(-0.4232233313872156) q[3];
cx q[2],q[3];
ry(0.28756380218112293) q[0];
ry(-3.050153743488726) q[2];
cx q[0],q[2];
ry(1.087725442606113) q[0];
ry(2.38308925997123) q[2];
cx q[0],q[2];
ry(-1.0773479525772913) q[1];
ry(-0.1310740657824875) q[3];
cx q[1],q[3];
ry(1.6210259632921482) q[1];
ry(1.0912161404268625) q[3];
cx q[1],q[3];
ry(1.3475638347927328) q[0];
ry(0.6316527908155134) q[1];
cx q[0],q[1];
ry(-2.04094546778635) q[0];
ry(1.9807216459008155) q[1];
cx q[0],q[1];
ry(-0.5779928942768724) q[2];
ry(2.5339207277338294) q[3];
cx q[2],q[3];
ry(-0.38881653642555847) q[2];
ry(-0.17347966590540764) q[3];
cx q[2],q[3];
ry(-3.065379782231571) q[0];
ry(-1.8275207290271935) q[2];
cx q[0],q[2];
ry(-1.3075335647092015) q[0];
ry(2.8468808067829796) q[2];
cx q[0],q[2];
ry(-0.8731238262038721) q[1];
ry(-1.510902291279086) q[3];
cx q[1],q[3];
ry(-1.5657276387896115) q[1];
ry(1.3682355223838132) q[3];
cx q[1],q[3];
ry(-0.8138478928309089) q[0];
ry(1.1269570160489693) q[1];
cx q[0],q[1];
ry(0.5947996903790047) q[0];
ry(-1.2527711905813979) q[1];
cx q[0],q[1];
ry(-2.9024163217551537) q[2];
ry(2.903092476933029) q[3];
cx q[2],q[3];
ry(0.6707695961308885) q[2];
ry(-2.600705376955783) q[3];
cx q[2],q[3];
ry(2.895226068620315) q[0];
ry(3.0340421813243412) q[2];
cx q[0],q[2];
ry(1.2452265628028618) q[0];
ry(-2.7662555748776376) q[2];
cx q[0],q[2];
ry(0.34175198587333455) q[1];
ry(-2.6911880689788177) q[3];
cx q[1],q[3];
ry(-0.16475323102619685) q[1];
ry(-0.9284216069691071) q[3];
cx q[1],q[3];
ry(1.37294202290406) q[0];
ry(-0.39192207449933036) q[1];
cx q[0],q[1];
ry(2.9006274393547162) q[0];
ry(-1.2508777535101079) q[1];
cx q[0],q[1];
ry(2.632191632414037) q[2];
ry(-1.9565931420717648) q[3];
cx q[2],q[3];
ry(2.8830580425364976) q[2];
ry(-2.8730194977145787) q[3];
cx q[2],q[3];
ry(-1.7373629001973547) q[0];
ry(-2.3282620822256663) q[2];
cx q[0],q[2];
ry(-1.3924416407778164) q[0];
ry(-0.3635827521052839) q[2];
cx q[0],q[2];
ry(-2.8216052440316965) q[1];
ry(2.949388138472785) q[3];
cx q[1],q[3];
ry(2.9505443016430593) q[1];
ry(-2.8122407400939347) q[3];
cx q[1],q[3];
ry(-1.9629652301095293) q[0];
ry(0.8592314604319116) q[1];
cx q[0],q[1];
ry(-1.4164188326762084) q[0];
ry(1.564531229851678) q[1];
cx q[0],q[1];
ry(-1.2697120527892114) q[2];
ry(0.45572629815575105) q[3];
cx q[2],q[3];
ry(-1.0074550556155448) q[2];
ry(1.3220336062915408) q[3];
cx q[2],q[3];
ry(1.13269464696) q[0];
ry(-0.9934802816347181) q[2];
cx q[0],q[2];
ry(-0.14835040126265486) q[0];
ry(1.758758255512776) q[2];
cx q[0],q[2];
ry(-1.2323591203287378) q[1];
ry(0.15671481718746294) q[3];
cx q[1],q[3];
ry(2.261657911073104) q[1];
ry(1.3754581584359498) q[3];
cx q[1],q[3];
ry(-1.4111255160461589) q[0];
ry(-2.1033115173111465) q[1];
cx q[0],q[1];
ry(1.198942045220562) q[0];
ry(2.086115375700323) q[1];
cx q[0],q[1];
ry(0.08840539838926069) q[2];
ry(-0.4808965100796243) q[3];
cx q[2],q[3];
ry(3.031364235619609) q[2];
ry(0.4612000648157651) q[3];
cx q[2],q[3];
ry(-2.307043549016653) q[0];
ry(1.9181813786389597) q[2];
cx q[0],q[2];
ry(2.7982321500562013) q[0];
ry(-1.0854624795931338) q[2];
cx q[0],q[2];
ry(-0.7792323636161456) q[1];
ry(1.374402607566811) q[3];
cx q[1],q[3];
ry(2.60732812968058) q[1];
ry(-1.2978118417635596) q[3];
cx q[1],q[3];
ry(-2.281266959664835) q[0];
ry(1.459828723300327) q[1];
cx q[0],q[1];
ry(-1.871819788560492) q[0];
ry(1.9299664209871044) q[1];
cx q[0],q[1];
ry(0.3808673855179796) q[2];
ry(-1.9370216858940974) q[3];
cx q[2],q[3];
ry(2.919178413347165) q[2];
ry(-1.7993225069480934) q[3];
cx q[2],q[3];
ry(0.5634477836927989) q[0];
ry(2.5994811095222548) q[2];
cx q[0],q[2];
ry(2.503202800834929) q[0];
ry(-1.2921462329471718) q[2];
cx q[0],q[2];
ry(-0.42427539217607624) q[1];
ry(0.49440257494843587) q[3];
cx q[1],q[3];
ry(1.151165957425816) q[1];
ry(-0.06580020132783489) q[3];
cx q[1],q[3];
ry(0.7826138561611007) q[0];
ry(-0.5998092204302298) q[1];
cx q[0],q[1];
ry(-2.964430373373427) q[0];
ry(1.342683018760222) q[1];
cx q[0],q[1];
ry(2.1483343251578586) q[2];
ry(0.8251856741909789) q[3];
cx q[2],q[3];
ry(-1.168677685674404) q[2];
ry(1.7555994051464185) q[3];
cx q[2],q[3];
ry(-1.4976269681712084) q[0];
ry(0.3447982747844991) q[2];
cx q[0],q[2];
ry(1.7904746181073088) q[0];
ry(0.3175908021026592) q[2];
cx q[0],q[2];
ry(-0.057661801282180694) q[1];
ry(1.183373699180315) q[3];
cx q[1],q[3];
ry(-1.042397622704681) q[1];
ry(-0.4131271238222425) q[3];
cx q[1],q[3];
ry(0.6900926965823121) q[0];
ry(-1.6217519662949575) q[1];
cx q[0],q[1];
ry(2.3106637099242895) q[0];
ry(2.2918098413129133) q[1];
cx q[0],q[1];
ry(-2.2464062086239807) q[2];
ry(0.4876473256198211) q[3];
cx q[2],q[3];
ry(2.295072668977993) q[2];
ry(-1.5511454420902326) q[3];
cx q[2],q[3];
ry(-1.980035109067079) q[0];
ry(-1.7258179190847893) q[2];
cx q[0],q[2];
ry(0.7475045369323299) q[0];
ry(1.7193363855755592) q[2];
cx q[0],q[2];
ry(-0.7815044814618807) q[1];
ry(0.36023485721731685) q[3];
cx q[1],q[3];
ry(-2.8671008561856164) q[1];
ry(-2.2993931905729608) q[3];
cx q[1],q[3];
ry(-0.5555213431845454) q[0];
ry(-0.4707178441908493) q[1];
cx q[0],q[1];
ry(-1.362099205025416) q[0];
ry(-2.016417970128561) q[1];
cx q[0],q[1];
ry(2.181688363226612) q[2];
ry(0.5292609126274055) q[3];
cx q[2],q[3];
ry(-2.958325932077689) q[2];
ry(0.08796059230381648) q[3];
cx q[2],q[3];
ry(-1.6255340165657246) q[0];
ry(2.4387885228870756) q[2];
cx q[0],q[2];
ry(-2.3856734879627433) q[0];
ry(-0.5485085659833265) q[2];
cx q[0],q[2];
ry(-0.2467590662282245) q[1];
ry(-2.7802048914647166) q[3];
cx q[1],q[3];
ry(-0.004783999671759625) q[1];
ry(-0.1293562272539429) q[3];
cx q[1],q[3];
ry(-2.7805653703911797) q[0];
ry(-1.8703158450809871) q[1];
cx q[0],q[1];
ry(0.6011312922194119) q[0];
ry(1.2628028582808515) q[1];
cx q[0],q[1];
ry(1.7528152329478508) q[2];
ry(-1.2465320825249764) q[3];
cx q[2],q[3];
ry(2.5780871678357165) q[2];
ry(-1.367776372227004) q[3];
cx q[2],q[3];
ry(1.8809903210890289) q[0];
ry(-2.493838516473184) q[2];
cx q[0],q[2];
ry(3.020235681671882) q[0];
ry(3.1137003756078334) q[2];
cx q[0],q[2];
ry(1.5759665192136696) q[1];
ry(-0.6363949126792701) q[3];
cx q[1],q[3];
ry(0.9321126288082525) q[1];
ry(3.0466509766294436) q[3];
cx q[1],q[3];
ry(-1.1780720648191032) q[0];
ry(-2.2825777866035564) q[1];
cx q[0],q[1];
ry(1.9744099080691628) q[0];
ry(2.177302837798689) q[1];
cx q[0],q[1];
ry(-0.17728980196341038) q[2];
ry(1.5066312157370545) q[3];
cx q[2],q[3];
ry(-0.9409426905021918) q[2];
ry(0.767384898806422) q[3];
cx q[2],q[3];
ry(1.7766756362811917) q[0];
ry(2.1145957589625715) q[2];
cx q[0],q[2];
ry(2.57712640729335) q[0];
ry(0.13919626640866886) q[2];
cx q[0],q[2];
ry(1.8970852292865021) q[1];
ry(-0.2820023886313614) q[3];
cx q[1],q[3];
ry(-0.4368337648324668) q[1];
ry(0.26629798176903385) q[3];
cx q[1],q[3];
ry(-2.171285998988412) q[0];
ry(0.6851331953913844) q[1];
cx q[0],q[1];
ry(2.1033422475388983) q[0];
ry(0.6379087659591386) q[1];
cx q[0],q[1];
ry(-2.199958434646278) q[2];
ry(-1.7721518901200382) q[3];
cx q[2],q[3];
ry(-3.038707103224903) q[2];
ry(1.8579213537044144) q[3];
cx q[2],q[3];
ry(1.595091610503668) q[0];
ry(-2.1811946252862247) q[2];
cx q[0],q[2];
ry(1.7908613149079127) q[0];
ry(-1.7859094234910873) q[2];
cx q[0],q[2];
ry(2.9312517185887645) q[1];
ry(-1.4966129317967078) q[3];
cx q[1],q[3];
ry(-3.0569637584597418) q[1];
ry(0.9464475118929915) q[3];
cx q[1],q[3];
ry(1.1412543001369226) q[0];
ry(-2.3043240477904448) q[1];
cx q[0],q[1];
ry(-1.3125691338664822) q[0];
ry(1.485268752941038) q[1];
cx q[0],q[1];
ry(-1.248641941785559) q[2];
ry(-2.6994615314675072) q[3];
cx q[2],q[3];
ry(0.6857159051373823) q[2];
ry(-0.9501176331068293) q[3];
cx q[2],q[3];
ry(-0.49947266680758157) q[0];
ry(0.025319292897675538) q[2];
cx q[0],q[2];
ry(1.33474697776459) q[0];
ry(2.4853865618598587) q[2];
cx q[0],q[2];
ry(-2.589845408602166) q[1];
ry(-1.278264957512178) q[3];
cx q[1],q[3];
ry(-2.987084039375652) q[1];
ry(1.3274507007278917) q[3];
cx q[1],q[3];
ry(-2.243032102564406) q[0];
ry(-1.8994651284511186) q[1];
cx q[0],q[1];
ry(1.653650091404883) q[0];
ry(-1.1304809285394084) q[1];
cx q[0],q[1];
ry(-1.4916316538391627) q[2];
ry(-0.900928866552932) q[3];
cx q[2],q[3];
ry(2.6734875408109775) q[2];
ry(0.6482575267168693) q[3];
cx q[2],q[3];
ry(2.8593515002148746) q[0];
ry(1.9774818326837413) q[2];
cx q[0],q[2];
ry(-2.893208642748779) q[0];
ry(1.5796863018544856) q[2];
cx q[0],q[2];
ry(1.5044609400623663) q[1];
ry(-0.4085981924016435) q[3];
cx q[1],q[3];
ry(-0.3504705294747801) q[1];
ry(0.6635470060078825) q[3];
cx q[1],q[3];
ry(-0.0695235804782044) q[0];
ry(-2.766874190422521) q[1];
cx q[0],q[1];
ry(2.1978886943197002) q[0];
ry(1.3452553230366275) q[1];
cx q[0],q[1];
ry(-0.33208602854590913) q[2];
ry(-2.2239588738505702) q[3];
cx q[2],q[3];
ry(-2.8140329716103163) q[2];
ry(2.9658288983319108) q[3];
cx q[2],q[3];
ry(-2.5796710506791607) q[0];
ry(-0.07212173876401184) q[2];
cx q[0],q[2];
ry(3.014849986127672) q[0];
ry(1.220411323476367) q[2];
cx q[0],q[2];
ry(2.714480368992116) q[1];
ry(3.1044075293156) q[3];
cx q[1],q[3];
ry(-1.9003546465465697) q[1];
ry(-1.5572217072137589) q[3];
cx q[1],q[3];
ry(2.289392594030883) q[0];
ry(-2.801168037588188) q[1];
cx q[0],q[1];
ry(0.11439898229140866) q[0];
ry(2.794020775118111) q[1];
cx q[0],q[1];
ry(-0.3194708077724427) q[2];
ry(-1.4691340029194544) q[3];
cx q[2],q[3];
ry(0.5318251862749054) q[2];
ry(-1.1431128212521982) q[3];
cx q[2],q[3];
ry(-0.9500109465213855) q[0];
ry(-2.46585852603056) q[2];
cx q[0],q[2];
ry(-1.220256478387177) q[0];
ry(1.2900046685571045) q[2];
cx q[0],q[2];
ry(-2.8743524441949484) q[1];
ry(3.0854629642835554) q[3];
cx q[1],q[3];
ry(2.7144333087500527) q[1];
ry(-1.3774117784983002) q[3];
cx q[1],q[3];
ry(-1.6097514883547248) q[0];
ry(-1.6154440803482952) q[1];
cx q[0],q[1];
ry(-1.593870118880809) q[0];
ry(0.18950697767639516) q[1];
cx q[0],q[1];
ry(2.238551629092804) q[2];
ry(2.971009455330545) q[3];
cx q[2],q[3];
ry(1.4436190286293054) q[2];
ry(0.4916841338594846) q[3];
cx q[2],q[3];
ry(0.24135605435598692) q[0];
ry(-0.8026377860095133) q[2];
cx q[0],q[2];
ry(2.8460682023051067) q[0];
ry(-0.4632858400023109) q[2];
cx q[0],q[2];
ry(-1.373049370931596) q[1];
ry(2.313473730345218) q[3];
cx q[1],q[3];
ry(0.06951788211714376) q[1];
ry(-0.6582848894031474) q[3];
cx q[1],q[3];
ry(-0.09430227953225234) q[0];
ry(-1.0223484391295095) q[1];
cx q[0],q[1];
ry(2.4402582233564942) q[0];
ry(-2.101724116411255) q[1];
cx q[0],q[1];
ry(-0.7320450163883315) q[2];
ry(1.5500836873494905) q[3];
cx q[2],q[3];
ry(-2.210795487518603) q[2];
ry(2.8945793637140813) q[3];
cx q[2],q[3];
ry(0.38336429515424353) q[0];
ry(1.0566326440920863) q[2];
cx q[0],q[2];
ry(1.771924427852832) q[0];
ry(0.493465543432249) q[2];
cx q[0],q[2];
ry(-2.5116037639067854) q[1];
ry(1.5867063419976903) q[3];
cx q[1],q[3];
ry(1.93755947578029) q[1];
ry(2.127767423974722) q[3];
cx q[1],q[3];
ry(2.423125381077217) q[0];
ry(-2.3007702252189084) q[1];
cx q[0],q[1];
ry(-0.6392073680294644) q[0];
ry(0.735566936257819) q[1];
cx q[0],q[1];
ry(1.035963388631266) q[2];
ry(-0.04450613584705178) q[3];
cx q[2],q[3];
ry(-1.77548377290585) q[2];
ry(-1.6614302114387494) q[3];
cx q[2],q[3];
ry(-2.7463309745668054) q[0];
ry(-2.7882376994595846) q[2];
cx q[0],q[2];
ry(-1.9164768639751744) q[0];
ry(-0.5440752704760304) q[2];
cx q[0],q[2];
ry(-0.612202159561118) q[1];
ry(-0.5436426242838177) q[3];
cx q[1],q[3];
ry(1.7997746610711465) q[1];
ry(0.5546550062085666) q[3];
cx q[1],q[3];
ry(1.126029443459033) q[0];
ry(-0.35873178430419017) q[1];
cx q[0],q[1];
ry(1.2871936192980014) q[0];
ry(0.7919594420435911) q[1];
cx q[0],q[1];
ry(0.6126681484206974) q[2];
ry(-1.409070984112703) q[3];
cx q[2],q[3];
ry(0.4182846446479349) q[2];
ry(-0.24725278297854025) q[3];
cx q[2],q[3];
ry(-2.676801676256769) q[0];
ry(1.5457413664501765) q[2];
cx q[0],q[2];
ry(-1.794375845349365) q[0];
ry(1.8872407170105983) q[2];
cx q[0],q[2];
ry(-1.14652620820278) q[1];
ry(2.511075802989576) q[3];
cx q[1],q[3];
ry(0.16181109151505918) q[1];
ry(2.8386321293251147) q[3];
cx q[1],q[3];
ry(-0.3196393447356618) q[0];
ry(0.5259246118055003) q[1];
cx q[0],q[1];
ry(2.922140111088255) q[0];
ry(0.4905780495964621) q[1];
cx q[0],q[1];
ry(2.1218489577611637) q[2];
ry(0.4753372233704705) q[3];
cx q[2],q[3];
ry(2.213120814166092) q[2];
ry(-2.336735271322153) q[3];
cx q[2],q[3];
ry(0.4957193008033995) q[0];
ry(1.3075458595411613) q[2];
cx q[0],q[2];
ry(-2.413311823562981) q[0];
ry(-1.0685770155050704) q[2];
cx q[0],q[2];
ry(1.838693167830154) q[1];
ry(-1.9027940057225952) q[3];
cx q[1],q[3];
ry(-0.1285515214197392) q[1];
ry(-0.9709568859669391) q[3];
cx q[1],q[3];
ry(-2.079814247723868) q[0];
ry(3.134670249396196) q[1];
cx q[0],q[1];
ry(-1.956818183247992) q[0];
ry(1.2182109499685385) q[1];
cx q[0],q[1];
ry(-3.079740706605847) q[2];
ry(2.7331227797907665) q[3];
cx q[2],q[3];
ry(2.843278273413214) q[2];
ry(0.8013101660427812) q[3];
cx q[2],q[3];
ry(0.4199807750068256) q[0];
ry(-2.710679408433377) q[2];
cx q[0],q[2];
ry(-2.698249742431487) q[0];
ry(-1.393450060734484) q[2];
cx q[0],q[2];
ry(2.5635910283925747) q[1];
ry(0.23701362887582622) q[3];
cx q[1],q[3];
ry(3.119935558239403) q[1];
ry(-2.0477545323825117) q[3];
cx q[1],q[3];
ry(-2.8719428995057634) q[0];
ry(-0.42537840985656494) q[1];
cx q[0],q[1];
ry(-1.958650135613166) q[0];
ry(-1.030187230866613) q[1];
cx q[0],q[1];
ry(3.0299627196006393) q[2];
ry(1.2144842725113767) q[3];
cx q[2],q[3];
ry(1.6741030657620632) q[2];
ry(2.396496081514544) q[3];
cx q[2],q[3];
ry(-1.8022387676046996) q[0];
ry(-1.9257095062768765) q[2];
cx q[0],q[2];
ry(0.7596401814239642) q[0];
ry(1.1265023127248162) q[2];
cx q[0],q[2];
ry(2.251873564989533) q[1];
ry(-0.5071296973460734) q[3];
cx q[1],q[3];
ry(0.7128466783132362) q[1];
ry(-2.330166479818213) q[3];
cx q[1],q[3];
ry(-2.618081595033429) q[0];
ry(-1.4786046566276925) q[1];
cx q[0],q[1];
ry(-2.1621059861028797) q[0];
ry(1.085742278626662) q[1];
cx q[0],q[1];
ry(-0.00986573252791849) q[2];
ry(1.2707976185187542) q[3];
cx q[2],q[3];
ry(2.480945072187645) q[2];
ry(2.4900665284004173) q[3];
cx q[2],q[3];
ry(0.30012879898532363) q[0];
ry(2.131453393522144) q[2];
cx q[0],q[2];
ry(2.09628729101074) q[0];
ry(2.3752125151534087) q[2];
cx q[0],q[2];
ry(0.15008134388517907) q[1];
ry(-2.2152665922160777) q[3];
cx q[1],q[3];
ry(-0.8165483279279906) q[1];
ry(1.0155377473588318) q[3];
cx q[1],q[3];
ry(0.23306589774563438) q[0];
ry(2.175744735359336) q[1];
cx q[0],q[1];
ry(-1.5537715670925778) q[0];
ry(-0.1658477007104846) q[1];
cx q[0],q[1];
ry(-1.9278480474278246) q[2];
ry(2.2986120287735825) q[3];
cx q[2],q[3];
ry(-3.1132018531704975) q[2];
ry(-2.697380784768101) q[3];
cx q[2],q[3];
ry(-0.8245118408835753) q[0];
ry(2.588350884643769) q[2];
cx q[0],q[2];
ry(1.201248353147724) q[0];
ry(1.4367356782775016) q[2];
cx q[0],q[2];
ry(1.6824297442296123) q[1];
ry(-1.3330790107334878) q[3];
cx q[1],q[3];
ry(-1.17555464365733) q[1];
ry(-0.7164241999261725) q[3];
cx q[1],q[3];
ry(-1.3789433346702467) q[0];
ry(1.4800644525897417) q[1];
cx q[0],q[1];
ry(2.534904602256677) q[0];
ry(2.975431989380649) q[1];
cx q[0],q[1];
ry(0.1595650684513358) q[2];
ry(2.1312672121533875) q[3];
cx q[2],q[3];
ry(1.9558950819646717) q[2];
ry(2.0210866362192714) q[3];
cx q[2],q[3];
ry(-2.1624062945085987) q[0];
ry(2.1901578866614777) q[2];
cx q[0],q[2];
ry(-1.9508030992758922) q[0];
ry(-0.6255229368481566) q[2];
cx q[0],q[2];
ry(0.8747111104839967) q[1];
ry(0.13889096084284078) q[3];
cx q[1],q[3];
ry(-1.401691831115157) q[1];
ry(1.4356025412282738) q[3];
cx q[1],q[3];
ry(1.2375279202868512) q[0];
ry(-0.2523628758509471) q[1];
cx q[0],q[1];
ry(-2.159959281963321) q[0];
ry(-2.9521779090927827) q[1];
cx q[0],q[1];
ry(-0.028793916804894515) q[2];
ry(-0.4097028572680177) q[3];
cx q[2],q[3];
ry(1.2227320905277788) q[2];
ry(2.637127527073184) q[3];
cx q[2],q[3];
ry(-1.6183147598455596) q[0];
ry(1.745754660601996) q[2];
cx q[0],q[2];
ry(-1.9034905097291603) q[0];
ry(0.6587500124162659) q[2];
cx q[0],q[2];
ry(2.1973841668795124) q[1];
ry(-1.1656787508110398) q[3];
cx q[1],q[3];
ry(1.506943867151583) q[1];
ry(-0.6913992478154549) q[3];
cx q[1],q[3];
ry(-2.4130054986286416) q[0];
ry(-1.733427832392535) q[1];
cx q[0],q[1];
ry(0.7751638172042519) q[0];
ry(1.6774477785743702) q[1];
cx q[0],q[1];
ry(-2.389171920630217) q[2];
ry(0.32378198919707846) q[3];
cx q[2],q[3];
ry(-0.5455540575449392) q[2];
ry(2.275536717813928) q[3];
cx q[2],q[3];
ry(2.526461204430869) q[0];
ry(-1.5053755200646646) q[2];
cx q[0],q[2];
ry(0.3077555545193551) q[0];
ry(-0.2543277753423423) q[2];
cx q[0],q[2];
ry(-1.2864484326736978) q[1];
ry(2.275033946964739) q[3];
cx q[1],q[3];
ry(-1.6339438806171094) q[1];
ry(1.3517675529218103) q[3];
cx q[1],q[3];
ry(-1.7430684078072571) q[0];
ry(2.9762780265613835) q[1];
cx q[0],q[1];
ry(0.4634137562516255) q[0];
ry(1.45010485496014) q[1];
cx q[0],q[1];
ry(3.0153618453357653) q[2];
ry(0.3399757390607139) q[3];
cx q[2],q[3];
ry(1.821600421978176) q[2];
ry(-2.7171210046800103) q[3];
cx q[2],q[3];
ry(-0.22414039249135073) q[0];
ry(-2.300148410215486) q[2];
cx q[0],q[2];
ry(1.9166181359528456) q[0];
ry(-0.3382683896502794) q[2];
cx q[0],q[2];
ry(-2.7253537939191923) q[1];
ry(-1.9350642117958845) q[3];
cx q[1],q[3];
ry(-0.34742566818653575) q[1];
ry(-0.3237283669068196) q[3];
cx q[1],q[3];
ry(2.1676758706663133) q[0];
ry(-1.6425586173368796) q[1];
cx q[0],q[1];
ry(-1.3828843319394508) q[0];
ry(3.0212963738259924) q[1];
cx q[0],q[1];
ry(-1.699419500958749) q[2];
ry(-0.008957667550376181) q[3];
cx q[2],q[3];
ry(-2.1411354866342) q[2];
ry(-1.3270006546274766) q[3];
cx q[2],q[3];
ry(-1.8718884795978834) q[0];
ry(0.38217914101099776) q[2];
cx q[0],q[2];
ry(-3.0978787600248694) q[0];
ry(-1.783301139121638) q[2];
cx q[0],q[2];
ry(-2.731186567244289) q[1];
ry(2.067666935748285) q[3];
cx q[1],q[3];
ry(-1.3372108914601144) q[1];
ry(1.1429146390332103) q[3];
cx q[1],q[3];
ry(2.067309862178676) q[0];
ry(2.0158362037718507) q[1];
ry(-2.6369197356240623) q[2];
ry(0.1800688370588297) q[3];