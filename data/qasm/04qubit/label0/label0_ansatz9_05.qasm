OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.318844049918414) q[0];
ry(2.1194018131543695) q[1];
cx q[0],q[1];
ry(1.4261273633377087) q[0];
ry(-0.8875042245315935) q[1];
cx q[0],q[1];
ry(-0.5811751244336859) q[2];
ry(1.9040655411786696) q[3];
cx q[2],q[3];
ry(-0.17732344928237254) q[2];
ry(3.129473761866985) q[3];
cx q[2],q[3];
ry(-1.917579147369322) q[0];
ry(3.040746383288401) q[2];
cx q[0],q[2];
ry(0.48114015677620436) q[0];
ry(1.1837814078738491) q[2];
cx q[0],q[2];
ry(0.28898207913096563) q[1];
ry(-0.7281491980298566) q[3];
cx q[1],q[3];
ry(2.623985343138874) q[1];
ry(-1.2366223207149791) q[3];
cx q[1],q[3];
ry(-1.5485184274980162) q[0];
ry(0.3620340876732895) q[3];
cx q[0],q[3];
ry(0.5982886172175897) q[0];
ry(1.1470910762859767) q[3];
cx q[0],q[3];
ry(-1.8628667539244288) q[1];
ry(1.895561058199294) q[2];
cx q[1],q[2];
ry(1.5170082823455884) q[1];
ry(1.994224813253462) q[2];
cx q[1],q[2];
ry(-1.596106348266483) q[0];
ry(0.9747051690199301) q[1];
cx q[0],q[1];
ry(-1.8603841203632834) q[0];
ry(-1.25500005651919) q[1];
cx q[0],q[1];
ry(2.7116464193080096) q[2];
ry(-0.8015510910652343) q[3];
cx q[2],q[3];
ry(1.5947263282406015) q[2];
ry(1.1447220062135144) q[3];
cx q[2],q[3];
ry(2.6013332459326906) q[0];
ry(1.223508630251589) q[2];
cx q[0],q[2];
ry(-1.913635164598711) q[0];
ry(-2.3572807811625114) q[2];
cx q[0],q[2];
ry(0.8817434985174203) q[1];
ry(0.08897369097481089) q[3];
cx q[1],q[3];
ry(-0.9904744236832764) q[1];
ry(-1.672024543361706) q[3];
cx q[1],q[3];
ry(1.5172333096634152) q[0];
ry(-1.4759233903672158) q[3];
cx q[0],q[3];
ry(0.22174957156689246) q[0];
ry(-3.018336639687345) q[3];
cx q[0],q[3];
ry(3.048306143407104) q[1];
ry(-2.07202688654618) q[2];
cx q[1],q[2];
ry(-0.28147554872903346) q[1];
ry(-0.48620270314546543) q[2];
cx q[1],q[2];
ry(-0.23645003637554057) q[0];
ry(1.8334204894353165) q[1];
cx q[0],q[1];
ry(-0.5163684011603777) q[0];
ry(0.029796037315725773) q[1];
cx q[0],q[1];
ry(1.9726943079258268) q[2];
ry(0.8846899789839651) q[3];
cx q[2],q[3];
ry(0.5106467732681628) q[2];
ry(-1.9032379361177467) q[3];
cx q[2],q[3];
ry(-1.7422786472016338) q[0];
ry(0.10310245367745718) q[2];
cx q[0],q[2];
ry(1.4157188660439524) q[0];
ry(0.9622080064605366) q[2];
cx q[0],q[2];
ry(1.9245159585443101) q[1];
ry(2.9490424975534606) q[3];
cx q[1],q[3];
ry(2.7217840510287763) q[1];
ry(1.368003704189551) q[3];
cx q[1],q[3];
ry(0.5267284232939833) q[0];
ry(2.931479557066336) q[3];
cx q[0],q[3];
ry(-2.7155857539890156) q[0];
ry(-0.9376888705131591) q[3];
cx q[0],q[3];
ry(-1.0495770815573733) q[1];
ry(-0.1719940188404232) q[2];
cx q[1],q[2];
ry(-0.6192317725059248) q[1];
ry(-3.064087835716714) q[2];
cx q[1],q[2];
ry(-1.7342853968808167) q[0];
ry(-2.2205194907401156) q[1];
cx q[0],q[1];
ry(-1.4785468778016235) q[0];
ry(-1.366708816839193) q[1];
cx q[0],q[1];
ry(-2.197152115144462) q[2];
ry(-1.6646649577911523) q[3];
cx q[2],q[3];
ry(0.6516451095723639) q[2];
ry(-1.8146092142995753) q[3];
cx q[2],q[3];
ry(0.1620978597283349) q[0];
ry(1.6641554643920324) q[2];
cx q[0],q[2];
ry(-0.5790094580249863) q[0];
ry(-1.4852045252211492) q[2];
cx q[0],q[2];
ry(0.6225390163405429) q[1];
ry(0.7084005649646876) q[3];
cx q[1],q[3];
ry(0.8169535161032302) q[1];
ry(0.29047100594524533) q[3];
cx q[1],q[3];
ry(0.9373330414664539) q[0];
ry(-2.510327456649109) q[3];
cx q[0],q[3];
ry(0.32153911069616736) q[0];
ry(-0.5245488672590343) q[3];
cx q[0],q[3];
ry(-0.12353351828759195) q[1];
ry(-1.3450791482108437) q[2];
cx q[1],q[2];
ry(-0.11022367968210713) q[1];
ry(-1.8048857233137552) q[2];
cx q[1],q[2];
ry(1.7045515153065525) q[0];
ry(1.3394891326827054) q[1];
cx q[0],q[1];
ry(-1.4766650096810725) q[0];
ry(-1.7479354171504689) q[1];
cx q[0],q[1];
ry(2.979496283544453) q[2];
ry(-1.80859783451452) q[3];
cx q[2],q[3];
ry(-0.025193412501782753) q[2];
ry(-2.684370499391959) q[3];
cx q[2],q[3];
ry(-1.0770956488122134) q[0];
ry(0.9876765816340424) q[2];
cx q[0],q[2];
ry(2.537438691144992) q[0];
ry(2.4123357017211657) q[2];
cx q[0],q[2];
ry(3.035116686357948) q[1];
ry(-2.6579150080628358) q[3];
cx q[1],q[3];
ry(1.202102662482777) q[1];
ry(-0.02318627130473505) q[3];
cx q[1],q[3];
ry(-0.1430195311759599) q[0];
ry(-0.22290901470595076) q[3];
cx q[0],q[3];
ry(2.50652971158723) q[0];
ry(1.825387899203591) q[3];
cx q[0],q[3];
ry(1.2252246431423472) q[1];
ry(-2.7084315509185877) q[2];
cx q[1],q[2];
ry(-0.5087120310405568) q[1];
ry(-0.6083982210117318) q[2];
cx q[1],q[2];
ry(-0.902366672184346) q[0];
ry(-3.0337159289924394) q[1];
cx q[0],q[1];
ry(2.8418059106012317) q[0];
ry(-0.8844502441060733) q[1];
cx q[0],q[1];
ry(-2.7658276290068957) q[2];
ry(1.306598080339163) q[3];
cx q[2],q[3];
ry(0.9264973866088785) q[2];
ry(0.4164534804863823) q[3];
cx q[2],q[3];
ry(1.637301927466787) q[0];
ry(-2.970414273119598) q[2];
cx q[0],q[2];
ry(0.34922008003245025) q[0];
ry(0.35689359681518606) q[2];
cx q[0],q[2];
ry(-1.2671304688874736) q[1];
ry(0.475195481562988) q[3];
cx q[1],q[3];
ry(2.5286149244179477) q[1];
ry(-0.3121975346673116) q[3];
cx q[1],q[3];
ry(-2.5340437960819613) q[0];
ry(1.6392764118764216) q[3];
cx q[0],q[3];
ry(-0.17613613733227318) q[0];
ry(0.594895755617581) q[3];
cx q[0],q[3];
ry(0.7771547859781456) q[1];
ry(0.45140509322338573) q[2];
cx q[1],q[2];
ry(0.3866085536024389) q[1];
ry(-2.295258159592038) q[2];
cx q[1],q[2];
ry(-2.6601121174455566) q[0];
ry(-3.1058322239457983) q[1];
cx q[0],q[1];
ry(0.4456017053735033) q[0];
ry(2.013484570292513) q[1];
cx q[0],q[1];
ry(2.4623649089412547) q[2];
ry(-2.7414095091771435) q[3];
cx q[2],q[3];
ry(2.2212429667894167) q[2];
ry(-2.421054626942604) q[3];
cx q[2],q[3];
ry(1.1625505466049812) q[0];
ry(-1.8002130011152584) q[2];
cx q[0],q[2];
ry(-1.6096566636556435) q[0];
ry(-0.8375701621612673) q[2];
cx q[0],q[2];
ry(-2.02732895811458) q[1];
ry(-0.4702304159921642) q[3];
cx q[1],q[3];
ry(1.1981164372806017) q[1];
ry(-2.425991739918556) q[3];
cx q[1],q[3];
ry(2.826998956022454) q[0];
ry(-1.3855540071754273) q[3];
cx q[0],q[3];
ry(0.29706572430909617) q[0];
ry(1.0795499747126067) q[3];
cx q[0],q[3];
ry(0.6183569970061242) q[1];
ry(-1.0487875610221358) q[2];
cx q[1],q[2];
ry(-0.9705770934623228) q[1];
ry(0.46890335203483513) q[2];
cx q[1],q[2];
ry(2.814341291037004) q[0];
ry(2.6357162175812583) q[1];
cx q[0],q[1];
ry(3.091997833821788) q[0];
ry(-2.367756126689802) q[1];
cx q[0],q[1];
ry(-2.2700099352963186) q[2];
ry(-0.11179481830940625) q[3];
cx q[2],q[3];
ry(0.11333338026948822) q[2];
ry(0.47379004549149695) q[3];
cx q[2],q[3];
ry(-0.450857798131997) q[0];
ry(1.536536639616749) q[2];
cx q[0],q[2];
ry(2.8527747711878444) q[0];
ry(-2.5008985965909507) q[2];
cx q[0],q[2];
ry(-2.2513337431478186) q[1];
ry(-3.061669609638505) q[3];
cx q[1],q[3];
ry(1.2083180711356718) q[1];
ry(0.0866676002049406) q[3];
cx q[1],q[3];
ry(1.08517844535446) q[0];
ry(2.013848111041125) q[3];
cx q[0],q[3];
ry(-0.7039366434843584) q[0];
ry(-3.009892201820742) q[3];
cx q[0],q[3];
ry(2.736218007999538) q[1];
ry(-1.8741406153642277) q[2];
cx q[1],q[2];
ry(-0.531564324550371) q[1];
ry(-0.3649933974928587) q[2];
cx q[1],q[2];
ry(-1.5722440776872444) q[0];
ry(0.9530644473707462) q[1];
ry(0.6630061721707149) q[2];
ry(-1.288179469296232) q[3];