OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.6421365046691392) q[0];
rz(-1.4739683021262113) q[0];
ry(-1.4039397862994627) q[1];
rz(-0.3881635785408165) q[1];
ry(2.663841103592284) q[2];
rz(2.1472995468029885) q[2];
ry(-2.738131944861697) q[3];
rz(-0.2831778634074787) q[3];
ry(-3.140925012320817) q[4];
rz(-0.7443474410494133) q[4];
ry(-3.1328977595188974) q[5];
rz(2.9614709037883595) q[5];
ry(0.0562473661883649) q[6];
rz(2.381350185770034) q[6];
ry(0.23509826011024249) q[7];
rz(-0.10085303365883061) q[7];
ry(3.1412898687405018) q[8];
rz(0.6100568651082989) q[8];
ry(0.004080599448037341) q[9];
rz(1.1232574977892655) q[9];
ry(-1.3557649724631207) q[10];
rz(-2.412635205787355) q[10];
ry(-1.5631599566502379) q[11];
rz(-3.0584474247949043) q[11];
ry(-0.00319128299920024) q[12];
rz(2.1173718989095223) q[12];
ry(3.1384779000597405) q[13];
rz(0.4856817488484521) q[13];
ry(0.4239984247635675) q[14];
rz(0.23639008785846394) q[14];
ry(-2.5416740806593743) q[15];
rz(-3.1387267673005015) q[15];
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
ry(-0.1632735232768372) q[0];
rz(1.6534283954635338) q[0];
ry(0.16619506818485671) q[1];
rz(-0.9704883278022547) q[1];
ry(0.39799643606502144) q[2];
rz(-2.3043699080103623) q[2];
ry(2.991317060067247) q[3];
rz(0.6624689397454463) q[3];
ry(-1.4544179915138153) q[4];
rz(-1.7189469727934412) q[4];
ry(2.5190903606135713) q[5];
rz(-0.06432074155183805) q[5];
ry(1.532184614618692) q[6];
rz(-1.5388015038228038) q[6];
ry(1.7906816319160974) q[7];
rz(1.5448105626229376) q[7];
ry(-1.286874348513028) q[8];
rz(1.3235075944650614) q[8];
ry(2.153971794325629) q[9];
rz(1.1151057120704648) q[9];
ry(3.0266671743958553) q[10];
rz(2.4006682527484395) q[10];
ry(-1.4876275466043907) q[11];
rz(-1.4486611925054387) q[11];
ry(0.609282232354137) q[12];
rz(0.282375775411519) q[12];
ry(-1.6808164275269795) q[13];
rz(1.22116201099887) q[13];
ry(2.8046913354716985) q[14];
rz(-2.501707021692851) q[14];
ry(2.5888818398983706) q[15];
rz(1.769463613940659) q[15];
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
ry(1.6755010769906873) q[0];
rz(1.6181031122876197) q[0];
ry(-2.2773369837125785) q[1];
rz(-2.7778280240062054) q[1];
ry(-3.141496690942691) q[2];
rz(-1.8890751158473231) q[2];
ry(0.001269834345633612) q[3];
rz(-0.7101763986497797) q[3];
ry(1.5748750195702899) q[4];
rz(3.1411928186621836) q[4];
ry(1.5677394421034327) q[5];
rz(0.0038442705008361507) q[5];
ry(1.5705687054923354) q[6];
rz(1.5860863661568665) q[6];
ry(-1.5706386008651574) q[7];
rz(1.5555695595601327) q[7];
ry(-1.1451295493396696) q[8];
rz(-2.7359617779925567) q[8];
ry(0.9069196597565616) q[9];
rz(-2.5661468163765835) q[9];
ry(3.141388121332642) q[10];
rz(0.2761316860880714) q[10];
ry(-3.140991018421969) q[11];
rz(2.6555827897297726) q[11];
ry(3.0753843698739773) q[12];
rz(1.681389189853438) q[12];
ry(0.868410628130442) q[13];
rz(1.6158133049852532) q[13];
ry(-0.0602471034309513) q[14];
rz(0.13091205191475108) q[14];
ry(3.1329973728370546) q[15];
rz(-1.8018133962220029) q[15];
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
ry(-2.8210867765132375) q[0];
rz(1.1436514312222754) q[0];
ry(1.5834926629430288) q[1];
rz(-0.8611404773650442) q[1];
ry(-2.0644757416476974e-05) q[2];
rz(1.4363578411319748) q[2];
ry(0.0001251741701465134) q[3];
rz(-0.6768608867295196) q[3];
ry(1.5681880910613624) q[4];
rz(0.3574076781609108) q[4];
ry(1.5668083735221858) q[5];
rz(-2.6025593550094492) q[5];
ry(1.5719835407387333) q[6];
rz(-3.12813493488286) q[6];
ry(1.57234446456844) q[7];
rz(3.1402002893752883) q[7];
ry(-1.2597837924679745) q[8];
rz(-1.598398522180616) q[8];
ry(0.9959772471436759) q[9];
rz(-1.836444488193789) q[9];
ry(-0.004469073551873315) q[10];
rz(-2.032343830150325) q[10];
ry(0.005273438974363168) q[11];
rz(2.2278550833365585) q[11];
ry(0.2884521154628814) q[12];
rz(2.821469898477794) q[12];
ry(1.224117932630514) q[13];
rz(0.21420293593730386) q[13];
ry(-1.12088203939887) q[14];
rz(1.872889082291545) q[14];
ry(-3.1306999533261544) q[15];
rz(-3.0775906655060443) q[15];
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
ry(-1.5903674918500093) q[0];
rz(0.2345502432123352) q[0];
ry(-2.53133452860475) q[1];
rz(0.3015867162876033) q[1];
ry(2.979244972608643) q[2];
rz(0.9424749412582891) q[2];
ry(1.9160713444993986) q[3];
rz(1.5134065725698687) q[3];
ry(0.2424882421435562) q[4];
rz(-2.698518955649646) q[4];
ry(-0.30594228501334175) q[5];
rz(0.3150905545922473) q[5];
ry(1.7411428607205064) q[6];
rz(1.088131006757318) q[6];
ry(-1.2713330419279218) q[7];
rz(1.492742377420758) q[7];
ry(-1.9737838214107128) q[8];
rz(-2.749325217844837) q[8];
ry(0.8423402646365856) q[9];
rz(2.3103313454841987) q[9];
ry(1.4165052054047482) q[10];
rz(1.601230999613577) q[10];
ry(0.020123030505299233) q[11];
rz(-1.0903304688490554) q[11];
ry(3.1242505032690464) q[12];
rz(-1.4561582655068404) q[12];
ry(-0.04060660005094797) q[13];
rz(1.33640978869487) q[13];
ry(-0.29068160157503703) q[14];
rz(1.2426020609921578) q[14];
ry(-0.004852506734041917) q[15];
rz(-1.9407481842432617) q[15];
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
ry(3.0660067454716984) q[0];
rz(0.4740523635190108) q[0];
ry(0.2859376123191337) q[1];
rz(1.5495323142826412) q[1];
ry(-3.1409950767697703) q[2];
rz(-0.030891043815662256) q[2];
ry(3.140577939090165) q[3];
rz(1.1307391785214904) q[3];
ry(3.1405291635725248) q[4];
rz(-0.4185853432787416) q[4];
ry(-3.140658682629122) q[5];
rz(-0.28395253254943986) q[5];
ry(2.9349155534650415) q[6];
rz(1.973765371016357) q[6];
ry(1.528423692482333) q[7];
rz(-1.4202700803744719) q[7];
ry(-3.1387965703619405) q[8];
rz(0.7077270177312887) q[8];
ry(-0.0012213476354601127) q[9];
rz(-1.3974488018474043) q[9];
ry(1.9587746833311703) q[10];
rz(-2.0922943268223175) q[10];
ry(-2.0005124062914184) q[11];
rz(1.247125767023545) q[11];
ry(-1.1322299428379416) q[12];
rz(1.969617728726985) q[12];
ry(1.317440868563785) q[13];
rz(-2.148987264478058) q[13];
ry(1.1985075913643914) q[14];
rz(0.9080596833170249) q[14];
ry(0.0028517440315349836) q[15];
rz(-0.14852274037519267) q[15];
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
ry(-1.5711797349618926) q[0];
rz(2.9612747014653125) q[0];
ry(-0.4506293887636126) q[1];
rz(-1.4591570717364062) q[1];
ry(0.538376062555213) q[2];
rz(-1.5322278198496262) q[2];
ry(-1.7331572937304554) q[3];
rz(-1.4535754426948477) q[3];
ry(1.526365231566859) q[4];
rz(-2.7642392927913546) q[4];
ry(-1.5098436839013472) q[5];
rz(-0.017795353288067486) q[5];
ry(-0.033600404371458545) q[6];
rz(0.23511343402085005) q[6];
ry(-0.20593752426274975) q[7];
rz(-0.6456851306167701) q[7];
ry(-0.003001086801509345) q[8];
rz(-0.19244865617213502) q[8];
ry(-3.141379751859934) q[9];
rz(1.3812965270555226) q[9];
ry(1.091060649950765) q[10];
rz(1.7929157998670044) q[10];
ry(2.3389830478988336) q[11];
rz(0.7798755490497173) q[11];
ry(1.9639900116098143) q[12];
rz(-1.75789219318493) q[12];
ry(0.6069545199492401) q[13];
rz(-0.9754251261304516) q[13];
ry(3.053067458465736) q[14];
rz(0.17773571379126304) q[14];
ry(0.03513735056650618) q[15];
rz(1.602594893450308) q[15];
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
ry(-2.8995511640762572) q[0];
rz(-2.662140274174345) q[0];
ry(1.9853793273972746) q[1];
rz(2.0352506570459843) q[1];
ry(0.000590385176087338) q[2];
rz(-2.8159755016809482) q[2];
ry(-0.0019959034912490235) q[3];
rz(-2.7022257796570517) q[3];
ry(1.5809047778626386) q[4];
rz(-1.58946380110929) q[4];
ry(-1.5972040426969913) q[5];
rz(-1.7747305148057513) q[5];
ry(-3.085900654119573) q[6];
rz(-0.5586435724054465) q[6];
ry(2.982189869985109) q[7];
rz(-0.7800761808672441) q[7];
ry(-0.000745075804658768) q[8];
rz(1.3135476191120699) q[8];
ry(3.140432136169936) q[9];
rz(2.5980349988793576) q[9];
ry(1.3337179341002425) q[10];
rz(2.6449062962915777) q[10];
ry(-2.522235251424767) q[11];
rz(-3.0736882167877155) q[11];
ry(0.41157466530024267) q[12];
rz(-1.197463602858735) q[12];
ry(0.30743715292340745) q[13];
rz(-2.100111672137729) q[13];
ry(-1.6002276556526818) q[14];
rz(1.6009906797806677) q[14];
ry(3.1364765310507092) q[15];
rz(-2.043747872527764) q[15];
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
ry(-1.332959680563902) q[0];
rz(0.5628147341861024) q[0];
ry(-1.5724758974472328) q[1];
rz(0.36189552755103266) q[1];
ry(3.1411763312700303) q[2];
rz(-0.34448733343183857) q[2];
ry(0.00031287419145620277) q[3];
rz(1.5782396349076138) q[3];
ry(-0.026892601530970034) q[4];
rz(2.5943734667399774) q[4];
ry(-3.0645173164831387) q[5];
rz(1.3467904039062775) q[5];
ry(-0.03653331514646258) q[6];
rz(-2.996816593343354) q[6];
ry(-1.1182023780851342) q[7];
rz(-1.0974229953470527) q[7];
ry(3.1408554618854243) q[8];
rz(1.8396632457936881) q[8];
ry(-3.1408232360427224) q[9];
rz(2.332591661974384) q[9];
ry(2.717200188940511) q[10];
rz(-2.341484773493008) q[10];
ry(1.0577833129272003) q[11];
rz(1.6998734048081348) q[11];
ry(3.1201313725609814) q[12];
rz(-0.7001694146277888) q[12];
ry(0.0020167972918025967) q[13];
rz(-0.9055790388338015) q[13];
ry(-1.2522586432145353) q[14];
rz(1.6800695848192695) q[14];
ry(-3.138429395300323) q[15];
rz(-2.0780280720026743) q[15];
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
ry(-2.8803105409916174) q[0];
rz(3.068787480542444) q[0];
ry(1.5266818629411631) q[1];
rz(1.2275496780460795) q[1];
ry(3.110884380021725) q[2];
rz(0.6719393938108533) q[2];
ry(-3.136911839729977) q[3];
rz(2.3112507222769447) q[3];
ry(2.8488663113696866) q[4];
rz(-2.2016369672123277) q[4];
ry(1.5426884379273615) q[5];
rz(-1.1654507987322797) q[5];
ry(1.236147269204839) q[6];
rz(1.5326040268018408) q[6];
ry(-0.6759705181662953) q[7];
rz(-2.065961002708537) q[7];
ry(3.0861474778263425) q[8];
rz(0.5383774061207847) q[8];
ry(-2.8879384806472603) q[9];
rz(-0.3451568372233665) q[9];
ry(-1.9043421806859513) q[10];
rz(-1.4718398930055903) q[10];
ry(2.1529577848728647) q[11];
rz(2.9172778223591873) q[11];
ry(-1.7845676874795184) q[12];
rz(2.5002138203153232) q[12];
ry(-0.7849800815105388) q[13];
rz(0.6781072843683139) q[13];
ry(-1.5754774283250832) q[14];
rz(-0.0007930956357830284) q[14];
ry(-3.09839601096931) q[15];
rz(-1.0909397796434384) q[15];
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
ry(-1.6622322908458949) q[0];
rz(0.8421366746580942) q[0];
ry(0.23130794542685615) q[1];
rz(-0.17500081237696463) q[1];
ry(-0.011021137994917218) q[2];
rz(2.6073794136461546) q[2];
ry(1.5989449084478216) q[3];
rz(2.3521128258863007) q[3];
ry(2.830924223770309) q[4];
rz(-1.2681142925723679) q[4];
ry(3.11591573421081) q[5];
rz(-0.06972458929331271) q[5];
ry(-1.5881867925790338) q[6];
rz(-1.5738774605215782) q[6];
ry(1.573003571309628) q[7];
rz(-0.4273937354084018) q[7];
ry(-0.0003220562646172328) q[8];
rz(-1.7984597351380256) q[8];
ry(-3.1415097090692834) q[9];
rz(-1.9030395210755942) q[9];
ry(0.17968126219299485) q[10];
rz(2.5590041828773287) q[10];
ry(-2.8702476422245686) q[11];
rz(1.9706507775022302) q[11];
ry(-2.0555436745090465) q[12];
rz(-2.094982150567102) q[12];
ry(-1.4626601626995321) q[13];
rz(-2.887952398158734) q[13];
ry(0.014750244906487355) q[14];
rz(-0.3179811139980239) q[14];
ry(3.138111714158879) q[15];
rz(-0.6373483526659491) q[15];
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
ry(-0.9456178962844084) q[0];
rz(-0.7284671076065035) q[0];
ry(-2.406405779442872) q[1];
rz(2.386303743437291) q[1];
ry(-1.6147180884237533) q[2];
rz(1.3096098760535957) q[2];
ry(-1.1980235520439697) q[3];
rz(-2.53628202753769) q[3];
ry(-3.1415159785388918) q[4];
rz(2.434657605173829) q[4];
ry(3.1376456293787314) q[5];
rz(2.79407422671444) q[5];
ry(2.0857058758721667) q[6];
rz(-0.6635716291311475) q[6];
ry(3.139537409905366) q[7];
rz(1.2352597696018044) q[7];
ry(3.1406819751033175) q[8];
rz(2.8550296923777387) q[8];
ry(0.00010704633600168737) q[9];
rz(-2.0772900507422154) q[9];
ry(-2.849427234458775) q[10];
rz(0.8355478128744734) q[10];
ry(-2.2422260939850753) q[11];
rz(-1.7464395849562155) q[11];
ry(-1.7706744490774209) q[12];
rz(-2.222831481864062) q[12];
ry(-0.20656141879167225) q[13];
rz(1.717862451069145) q[13];
ry(-3.1207635760223935) q[14];
rz(-1.3743433399767515) q[14];
ry(-3.1404533015717573) q[15];
rz(-1.707571690093315) q[15];
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
ry(3.1396984387232556) q[0];
rz(1.212481853120292) q[0];
ry(0.0059996234451800134) q[1];
rz(0.3396209733517134) q[1];
ry(0.0022496850690426244) q[2];
rz(-0.6628712584333728) q[2];
ry(3.1381597674470503) q[3];
rz(0.766091840117779) q[3];
ry(0.00017039402136731496) q[4];
rz(0.5081381706908202) q[4];
ry(-3.1413040188984493) q[5];
rz(-0.2918479667111453) q[5];
ry(0.07531987843075605) q[6];
rz(0.616466060910301) q[6];
ry(1.6632667482313392) q[7];
rz(0.6144170340721011) q[7];
ry(3.1398371573890835) q[8];
rz(3.0937421290710714) q[8];
ry(-0.00033463822380873457) q[9];
rz(1.0778935299950012) q[9];
ry(1.2953090538435177) q[10];
rz(0.0830112342387251) q[10];
ry(-1.7913820173766464) q[11];
rz(1.1852753424717495) q[11];
ry(-0.8418181674909622) q[12];
rz(-2.437913742699711) q[12];
ry(0.020062508321635697) q[13];
rz(1.5988039909974203) q[13];
ry(0.039392096914224524) q[14];
rz(-0.06077139912767304) q[14];
ry(-0.0098074602657503) q[15];
rz(-0.8686600084950706) q[15];
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
ry(2.046480362492451) q[0];
rz(2.56184266929887) q[0];
ry(-0.889974354161923) q[1];
rz(1.4379097044563631) q[1];
ry(3.088688495027932) q[2];
rz(0.6415873694404013) q[2];
ry(0.37880992970231997) q[3];
rz(-0.1483202461112265) q[3];
ry(3.1414037622046953) q[4];
rz(-1.2179840176781642) q[4];
ry(0.0004243970505036177) q[5];
rz(2.1450442089688977) q[5];
ry(-3.097021311828418) q[6];
rz(-1.6197102792647753) q[6];
ry(-0.03154840125654079) q[7];
rz(-2.168837972766614) q[7];
ry(-2.8886842785172133) q[8];
rz(3.030097854079545) q[8];
ry(-3.110591271877772) q[9];
rz(-0.11841198780693075) q[9];
ry(0.44534509184600934) q[10];
rz(0.8615632552098571) q[10];
ry(-0.5919590076817265) q[11];
rz(-1.5294661936959253) q[11];
ry(2.0306504633285902) q[12];
rz(1.1717336513003225) q[12];
ry(-1.5098302897845515) q[13];
rz(2.7265936210815105) q[13];
ry(-1.260977330161011) q[14];
rz(-1.1442928737843516) q[14];
ry(1.672401172998103) q[15];
rz(1.4269352933537776) q[15];
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
ry(-3.1397180636412574) q[0];
rz(-2.864275078360744) q[0];
ry(0.0024731072122472497) q[1];
rz(-1.1683771705431487) q[1];
ry(1.5677997343780299) q[2];
rz(-2.6437214322432556) q[2];
ry(1.5672501350852883) q[3];
rz(-0.4913704611727976) q[3];
ry(-0.0009245921771021486) q[4];
rz(0.725668801934213) q[4];
ry(-3.1413038758602503) q[5];
rz(-2.713747866214494) q[5];
ry(1.5458273492954238) q[6];
rz(-1.5829096906239044) q[6];
ry(1.6655336420595894) q[7];
rz(2.6683569440530466) q[7];
ry(-3.140957911508677) q[8];
rz(1.4683175386215823) q[8];
ry(-0.0007386486788325541) q[9];
rz(1.734148167195535) q[9];
ry(-0.009082356910561806) q[10];
rz(-2.103381730977454) q[10];
ry(-3.1353977211318815) q[11];
rz(1.4814265438733392) q[11];
ry(-0.0013977074669595524) q[12];
rz(-2.162448604812938) q[12];
ry(-3.1413047449792875) q[13];
rz(-1.993404325006085) q[13];
ry(0.08638934424224766) q[14];
rz(0.35202018946089897) q[14];
ry(0.26219075842801043) q[15];
rz(-1.6196163672617425) q[15];
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
ry(-1.1490473107902819) q[0];
rz(-1.5715382540388434) q[0];
ry(-1.70732559560371) q[1];
rz(1.5717233429667425) q[1];
ry(-2.0919699464443484) q[2];
rz(-0.7014085902919627) q[2];
ry(3.113013529006256) q[3];
rz(-0.539946860097007) q[3];
ry(1.5038722479755111) q[4];
rz(3.0790587033343133) q[4];
ry(0.14546964077086905) q[5];
rz(-1.907475653157145) q[5];
ry(1.3325931115636618) q[6];
rz(-1.6114034522220033) q[6];
ry(3.1305652961992245) q[7];
rz(-2.2100010423729035) q[7];
ry(-1.5771717321624665) q[8];
rz(3.115065477622452) q[8];
ry(1.560930802986174) q[9];
rz(-0.007292028864787447) q[9];
ry(-0.288311693585378) q[10];
rz(2.589110692972872) q[10];
ry(2.919706697109142) q[11];
rz(-1.493446020158805) q[11];
ry(1.499272785778111) q[12];
rz(3.076708979900112) q[12];
ry(-1.2619632282672069) q[13];
rz(-1.5369145071153143) q[13];
ry(-2.7108642925220767) q[14];
rz(2.428591337293766) q[14];
ry(0.10103476299614746) q[15];
rz(-2.947118252011365) q[15];