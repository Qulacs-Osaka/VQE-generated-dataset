OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.8124488793899354) q[0];
ry(0.15418937863213333) q[1];
cx q[0],q[1];
ry(1.1208818771356464) q[0];
ry(0.7189053518053741) q[1];
cx q[0],q[1];
ry(-0.20880369584398137) q[2];
ry(-1.661997785223806) q[3];
cx q[2],q[3];
ry(0.07601035159483907) q[2];
ry(-2.4258760063058857) q[3];
cx q[2],q[3];
ry(-2.354107369684183) q[0];
ry(-1.1520084714611178) q[2];
cx q[0],q[2];
ry(0.1787760046752584) q[0];
ry(0.890797116954352) q[2];
cx q[0],q[2];
ry(-0.49694764330479035) q[1];
ry(2.6060239984230162) q[3];
cx q[1],q[3];
ry(1.34290838933007) q[1];
ry(2.69635610941867) q[3];
cx q[1],q[3];
ry(-0.3163804297146662) q[0];
ry(-0.7924125772481242) q[3];
cx q[0],q[3];
ry(2.993153311326561) q[0];
ry(-2.1504609892409623) q[3];
cx q[0],q[3];
ry(-2.7392867732169046) q[1];
ry(-2.4173515471500884) q[2];
cx q[1],q[2];
ry(-0.20086101794822842) q[1];
ry(-0.2724023711088961) q[2];
cx q[1],q[2];
ry(-0.8685843251982535) q[0];
ry(-0.5219636290343201) q[1];
cx q[0],q[1];
ry(3.0051508794391393) q[0];
ry(-2.3487839400899415) q[1];
cx q[0],q[1];
ry(3.0876170373889082) q[2];
ry(-2.1782357157033903) q[3];
cx q[2],q[3];
ry(2.307413174523442) q[2];
ry(0.49856550151634565) q[3];
cx q[2],q[3];
ry(-2.019815239659963) q[0];
ry(-2.873594651261015) q[2];
cx q[0],q[2];
ry(-2.8945943647967187) q[0];
ry(-2.8963032815490406) q[2];
cx q[0],q[2];
ry(-2.1244699131084905) q[1];
ry(2.436786491024555) q[3];
cx q[1],q[3];
ry(-0.41817922416911574) q[1];
ry(0.7999643997824757) q[3];
cx q[1],q[3];
ry(-1.7754187404561668) q[0];
ry(2.240547082046696) q[3];
cx q[0],q[3];
ry(1.5744298610302077) q[0];
ry(1.0628763683452602) q[3];
cx q[0],q[3];
ry(2.295162258498586) q[1];
ry(-0.9111318801392219) q[2];
cx q[1],q[2];
ry(-2.229468995069499) q[1];
ry(-2.982336843961129) q[2];
cx q[1],q[2];
ry(-2.78366483744903) q[0];
ry(-2.201799757948817) q[1];
cx q[0],q[1];
ry(3.021681327654697) q[0];
ry(2.6341450361250027) q[1];
cx q[0],q[1];
ry(-1.8564183988095033) q[2];
ry(0.7998801988809916) q[3];
cx q[2],q[3];
ry(-2.9409175292211978) q[2];
ry(-0.8751628258631374) q[3];
cx q[2],q[3];
ry(2.3722002930371047) q[0];
ry(-0.8957357436309435) q[2];
cx q[0],q[2];
ry(0.0833211795721569) q[0];
ry(2.784058548737486) q[2];
cx q[0],q[2];
ry(-0.6432451733359702) q[1];
ry(-2.763074995708899) q[3];
cx q[1],q[3];
ry(0.8740820062856487) q[1];
ry(1.1025882060481789) q[3];
cx q[1],q[3];
ry(0.2806713361617881) q[0];
ry(0.5437437887049459) q[3];
cx q[0],q[3];
ry(-0.5848004150515447) q[0];
ry(0.8627587765184384) q[3];
cx q[0],q[3];
ry(2.460160315612422) q[1];
ry(0.9772072004098801) q[2];
cx q[1],q[2];
ry(-0.18979401688544947) q[1];
ry(-1.3156202599917906) q[2];
cx q[1],q[2];
ry(-2.158716523039855) q[0];
ry(0.6023126694346761) q[1];
cx q[0],q[1];
ry(1.0328701967103422) q[0];
ry(-1.429534465193292) q[1];
cx q[0],q[1];
ry(-2.677357664509222) q[2];
ry(2.625788044572296) q[3];
cx q[2],q[3];
ry(1.0313319570226227) q[2];
ry(1.5836406138046506) q[3];
cx q[2],q[3];
ry(-1.8275684436895883) q[0];
ry(-1.3122422978056578) q[2];
cx q[0],q[2];
ry(-2.0630040769434994) q[0];
ry(-1.644090427707562) q[2];
cx q[0],q[2];
ry(0.3152293551079177) q[1];
ry(1.1237084453588722) q[3];
cx q[1],q[3];
ry(0.08554491712230454) q[1];
ry(-1.6710683912611473) q[3];
cx q[1],q[3];
ry(-2.546178883718677) q[0];
ry(-0.6147999706193046) q[3];
cx q[0],q[3];
ry(-2.3864400952427096) q[0];
ry(-1.1750048219821985) q[3];
cx q[0],q[3];
ry(-0.27739290554102847) q[1];
ry(0.43104817737660317) q[2];
cx q[1],q[2];
ry(-1.0080748855871704) q[1];
ry(-0.27819040685092866) q[2];
cx q[1],q[2];
ry(-1.0765959177936661) q[0];
ry(-0.252275248998286) q[1];
cx q[0],q[1];
ry(-2.686891250614986) q[0];
ry(1.9362036589081848) q[1];
cx q[0],q[1];
ry(-0.5394599355457401) q[2];
ry(0.20020647600472669) q[3];
cx q[2],q[3];
ry(0.10977643033369322) q[2];
ry(-0.5539930819170278) q[3];
cx q[2],q[3];
ry(0.024110287078176494) q[0];
ry(1.9478519117578443) q[2];
cx q[0],q[2];
ry(2.3534488160802463) q[0];
ry(-0.13151233752549008) q[2];
cx q[0],q[2];
ry(-3.075316578061365) q[1];
ry(-3.0696904331636374) q[3];
cx q[1],q[3];
ry(1.1173977661067411) q[1];
ry(-0.01787378814783898) q[3];
cx q[1],q[3];
ry(3.0800347702300312) q[0];
ry(-2.380423124624562) q[3];
cx q[0],q[3];
ry(-2.695983290197331) q[0];
ry(2.285960219184498) q[3];
cx q[0],q[3];
ry(2.6349217553560758) q[1];
ry(-1.0817230570714298) q[2];
cx q[1],q[2];
ry(0.5703785974310636) q[1];
ry(-1.9794011587223357) q[2];
cx q[1],q[2];
ry(-1.6689666839776993) q[0];
ry(1.5929283819672166) q[1];
cx q[0],q[1];
ry(-2.798889130190707) q[0];
ry(3.0123902561674605) q[1];
cx q[0],q[1];
ry(-0.36254224229027027) q[2];
ry(-1.9011715188971858) q[3];
cx q[2],q[3];
ry(2.209868253256342) q[2];
ry(2.089325259436042) q[3];
cx q[2],q[3];
ry(0.9034013196693458) q[0];
ry(-1.6365629434469633) q[2];
cx q[0],q[2];
ry(-0.18494677698616258) q[0];
ry(1.757340738389402) q[2];
cx q[0],q[2];
ry(0.8651413709164464) q[1];
ry(2.368870641677338) q[3];
cx q[1],q[3];
ry(-2.698725421470857) q[1];
ry(-0.21253107536617044) q[3];
cx q[1],q[3];
ry(-0.21349844577939525) q[0];
ry(-2.757032035827485) q[3];
cx q[0],q[3];
ry(2.2079141270293023) q[0];
ry(0.25327682576402033) q[3];
cx q[0],q[3];
ry(-0.09594743141559103) q[1];
ry(-0.16770554288073214) q[2];
cx q[1],q[2];
ry(-0.3264161891553505) q[1];
ry(2.3815173561488163) q[2];
cx q[1],q[2];
ry(-0.49035162023983897) q[0];
ry(-1.408366652916687) q[1];
ry(-0.15037372860977943) q[2];
ry(2.9359586651972007) q[3];