OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.5327802765003007) q[0];
rz(2.057155745845996) q[0];
ry(1.0230182650159614) q[1];
rz(-1.7904968971760873) q[1];
ry(1.6734502863655898) q[2];
rz(-2.9105832563234673) q[2];
ry(-2.324902109301114) q[3];
rz(2.1257567930628083) q[3];
ry(0.8755681927127235) q[4];
rz(2.8799850869199433) q[4];
ry(1.2690353308277977) q[5];
rz(-1.4349502113847856) q[5];
ry(-2.761981201016882) q[6];
rz(-2.386046050513077) q[6];
ry(0.5526024924329471) q[7];
rz(2.7456248452954126) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.258627603293171) q[0];
rz(2.242022577252233) q[0];
ry(0.5871420358888164) q[1];
rz(1.5590565736302369) q[1];
ry(-2.3494638828631307) q[2];
rz(1.6598632678396585) q[2];
ry(2.1487164040788977) q[3];
rz(-0.9325302368890434) q[3];
ry(-1.2706968677962163) q[4];
rz(1.1160256738253935) q[4];
ry(-2.3754199574852377) q[5];
rz(1.3157721962551585) q[5];
ry(2.3294089214607565) q[6];
rz(2.1015082014755597) q[6];
ry(1.3114693211178718) q[7];
rz(1.1018386978190344) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.6294537604631127) q[0];
rz(-1.9218599994538899) q[0];
ry(-2.0940171054055248) q[1];
rz(-1.5381895632901053) q[1];
ry(-2.2001572769344744) q[2];
rz(-0.9226815211610571) q[2];
ry(0.3953514181292519) q[3];
rz(-2.853860299144394) q[3];
ry(2.34180497056674) q[4];
rz(-1.7654783168017474) q[4];
ry(2.5964912167297456) q[5];
rz(2.929958724328524) q[5];
ry(-2.8536692308017817) q[6];
rz(0.10316087601097265) q[6];
ry(-0.6327067554926532) q[7];
rz(-1.8630347240048728) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.5349725149228988) q[0];
rz(2.9383716733596543) q[0];
ry(-0.35691356368862337) q[1];
rz(-0.887122886841647) q[1];
ry(1.0003412503672635) q[2];
rz(2.9963899322642376) q[2];
ry(-2.1732986728657133) q[3];
rz(-0.9683147983347808) q[3];
ry(-2.433397800422862) q[4];
rz(-2.1733288949898197) q[4];
ry(1.9340260030583831) q[5];
rz(-2.772710814805343) q[5];
ry(-1.9868834928093133) q[6];
rz(-0.6251726627736822) q[6];
ry(-2.442647418019358) q[7];
rz(0.5792626140237731) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5903149777832768) q[0];
rz(1.0676685077042745) q[0];
ry(0.4393554278207789) q[1];
rz(-1.1038232273460578) q[1];
ry(-1.7111425872419632) q[2];
rz(2.7085635670734827) q[2];
ry(1.3355017672485303) q[3];
rz(-1.6246198493041608) q[3];
ry(-1.6830250323182225) q[4];
rz(0.4622130283805203) q[4];
ry(0.6138304034712485) q[5];
rz(0.9203753405983459) q[5];
ry(0.40774090569767485) q[6];
rz(2.3938246712311413) q[6];
ry(1.378413176572359) q[7];
rz(-1.7529897748474137) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.48175276391423877) q[0];
rz(-2.2446526869508143) q[0];
ry(1.081758173464543) q[1];
rz(-2.657261668460647) q[1];
ry(2.474648674266666) q[2];
rz(1.603110518429098) q[2];
ry(0.25903582360187904) q[3];
rz(-0.4689309135152149) q[3];
ry(-1.8923454755069926) q[4];
rz(1.7178699272056692) q[4];
ry(-0.421443684761064) q[5];
rz(-2.342961543627822) q[5];
ry(-1.2612695872604431) q[6];
rz(0.825741609098322) q[6];
ry(1.2933711828188112) q[7];
rz(-2.539183414726692) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.33130408837323794) q[0];
rz(2.0956931916750285) q[0];
ry(-0.9595248520442334) q[1];
rz(3.1133591057829912) q[1];
ry(0.7366005120418047) q[2];
rz(3.128548581366879) q[2];
ry(-0.7562656093163804) q[3];
rz(-1.1105091756311252) q[3];
ry(-0.5489959685157139) q[4];
rz(2.2659024811586237) q[4];
ry(-2.6951914242813504) q[5];
rz(1.0822407950377633) q[5];
ry(2.742329999460348) q[6];
rz(-2.533296848910094) q[6];
ry(-0.6535044111387767) q[7];
rz(-2.023784991709712) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.4333467627363996) q[0];
rz(-1.3849435694892895) q[0];
ry(2.857239157031831) q[1];
rz(2.3637713257673245) q[1];
ry(-1.4790100684005445) q[2];
rz(1.2258284766838674) q[2];
ry(-0.5771372641219168) q[3];
rz(-3.13689362896715) q[3];
ry(2.202820934538825) q[4];
rz(-2.797110014629136) q[4];
ry(0.8580235952741236) q[5];
rz(1.1216988132588357) q[5];
ry(-1.2697129734973869) q[6];
rz(-0.6917971948757841) q[6];
ry(-3.032151849254879) q[7];
rz(-1.9611372902778592) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.1684358960847794) q[0];
rz(-0.8761698558994143) q[0];
ry(-2.2532744420946758) q[1];
rz(-1.7767510390633694) q[1];
ry(1.208159775579737) q[2];
rz(-2.8813996273694915) q[2];
ry(1.252477380578849) q[3];
rz(0.9904155675732133) q[3];
ry(1.507713438578555) q[4];
rz(0.04117872116831113) q[4];
ry(-0.4137200709886875) q[5];
rz(-0.8566708225521166) q[5];
ry(-1.757985843022854) q[6];
rz(-1.1481157118878684) q[6];
ry(1.1096076941029276) q[7];
rz(-0.880149809743444) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.7063142210325364) q[0];
rz(-0.3391298728251613) q[0];
ry(-1.2179924329119778) q[1];
rz(-0.7809932222805731) q[1];
ry(1.4560561423935123) q[2];
rz(1.055622965337047) q[2];
ry(-1.2432567830861454) q[3];
rz(-1.0174533912667825) q[3];
ry(2.9500337854127796) q[4];
rz(2.94125559146617) q[4];
ry(2.721499240998243) q[5];
rz(2.0883820627322285) q[5];
ry(2.5208420066160864) q[6];
rz(0.12124481457244202) q[6];
ry(1.35822256256811) q[7];
rz(1.592612542305272) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.7083685743357675) q[0];
rz(-0.6272746425104189) q[0];
ry(-0.4325451772912636) q[1];
rz(-2.9790333197978236) q[1];
ry(2.979607740792134) q[2];
rz(2.7089104113569618) q[2];
ry(-0.6272733861603959) q[3];
rz(-1.5921229471654168) q[3];
ry(-0.7663259024617838) q[4];
rz(0.2963611759667051) q[4];
ry(2.829896921917369) q[5];
rz(2.671858463655524) q[5];
ry(1.004985966612404) q[6];
rz(2.4438905106954554) q[6];
ry(0.8740510734710147) q[7];
rz(0.44729968448205687) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.35939539892492345) q[0];
rz(2.699189274613841) q[0];
ry(1.330218730761229) q[1];
rz(-0.3869083097930428) q[1];
ry(-1.3717018724220305) q[2];
rz(3.08342838522426) q[2];
ry(-2.78167792967173) q[3];
rz(1.642426057972201) q[3];
ry(-0.8690062444731002) q[4];
rz(-2.2952998240698417) q[4];
ry(2.4187160104095526) q[5];
rz(-1.1893587040982883) q[5];
ry(2.6727013953608596) q[6];
rz(0.4849151222082102) q[6];
ry(-2.603225805067879) q[7];
rz(-0.4643880143993231) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.1552341153368022) q[0];
rz(-2.899185047209571) q[0];
ry(-0.35764142612504735) q[1];
rz(1.8973967224624344) q[1];
ry(1.3548954491435536) q[2];
rz(-1.9023303134599294) q[2];
ry(-1.3583668938919784) q[3];
rz(-0.12258172979185745) q[3];
ry(-2.0660625938414214) q[4];
rz(0.6283342915639031) q[4];
ry(-1.4778780844617643) q[5];
rz(-2.320023668435336) q[5];
ry(-2.7276211350542368) q[6];
rz(0.9628494634539119) q[6];
ry(0.19961401732788867) q[7];
rz(-2.7884206222528243) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.6854667972414928) q[0];
rz(1.0600120901506918) q[0];
ry(1.9217611970281987) q[1];
rz(-0.6345653337458739) q[1];
ry(-2.0119249145480054) q[2];
rz(2.8104410882805775) q[2];
ry(1.1183431051299182) q[3];
rz(2.167397868493338) q[3];
ry(1.9190529180835014) q[4];
rz(-2.6939934729359747) q[4];
ry(-1.1490969887537599) q[5];
rz(-0.05779390750498711) q[5];
ry(-1.6800015523459022) q[6];
rz(-2.3829250581144694) q[6];
ry(2.3122863806204554) q[7];
rz(0.3919264249748444) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.5321507890727997) q[0];
rz(-1.4040326134521797) q[0];
ry(-0.3664987820504946) q[1];
rz(2.409128365643283) q[1];
ry(-3.0541471133267883) q[2];
rz(-0.7380749508562046) q[2];
ry(2.5642296034460568) q[3];
rz(0.5822247325324357) q[3];
ry(2.296319385973679) q[4];
rz(-0.18294030811927706) q[4];
ry(1.3928538798623455) q[5];
rz(0.4421365161609048) q[5];
ry(1.345386744379583) q[6];
rz(-0.2474372780508469) q[6];
ry(-2.4893094669314078) q[7];
rz(-0.9547131512651773) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.239450793181004) q[0];
rz(-1.7919431224826083) q[0];
ry(1.9322893131478034) q[1];
rz(1.8329128447803236) q[1];
ry(2.9142630382365655) q[2];
rz(1.0292761678445306) q[2];
ry(1.6490843702291418) q[3];
rz(2.181534232440785) q[3];
ry(-0.3200520534838569) q[4];
rz(2.8124596259831685) q[4];
ry(-0.44444062970961706) q[5];
rz(0.3947868683127742) q[5];
ry(2.904378676247603) q[6];
rz(-1.7499101390853307) q[6];
ry(-1.9566586441188816) q[7];
rz(0.3816933823466239) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.4652268940676105) q[0];
rz(1.9128194682379616) q[0];
ry(1.9711590824331637) q[1];
rz(1.951036431081545) q[1];
ry(-0.8784889286835487) q[2];
rz(-0.6702847268078046) q[2];
ry(-1.104124416922179) q[3];
rz(2.5153626881415043) q[3];
ry(2.3642612224152457) q[4];
rz(1.8810439219144257) q[4];
ry(1.157643761748293) q[5];
rz(-1.368526254130938) q[5];
ry(-0.19249015846989426) q[6];
rz(-1.4708043429277744) q[6];
ry(-1.3818335086403637) q[7];
rz(0.6459969081156968) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.4021636491635714) q[0];
rz(-1.83589693384977) q[0];
ry(-2.433705056458889) q[1];
rz(2.6271478815260014) q[1];
ry(-2.550435530772425) q[2];
rz(2.448750409039387) q[2];
ry(-0.48243007716828784) q[3];
rz(2.507968162121508) q[3];
ry(-2.270971263191841) q[4];
rz(-0.5033488122689107) q[4];
ry(2.3304218230967333) q[5];
rz(-2.3269873645508286) q[5];
ry(-2.5932527087605375) q[6];
rz(0.7316101428773027) q[6];
ry(0.7060916808022055) q[7];
rz(-0.9652865170260094) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.3938440869270936) q[0];
rz(0.17813222434818826) q[0];
ry(-1.9043142753065991) q[1];
rz(2.688883268693581) q[1];
ry(-0.6622163803826008) q[2];
rz(2.9547001790538454) q[2];
ry(1.5911576927116613) q[3];
rz(1.9742214054895049) q[3];
ry(0.5261048288296823) q[4];
rz(1.2107900767286872) q[4];
ry(1.0190305490594511) q[5];
rz(2.4925731616980804) q[5];
ry(1.756892761553204) q[6];
rz(-1.1437531695736611) q[6];
ry(2.759490796002822) q[7];
rz(0.6841032282887616) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.6496616284586572) q[0];
rz(1.5465426037582988) q[0];
ry(2.685827584582785) q[1];
rz(2.2315875705200856) q[1];
ry(2.6285937079185984) q[2];
rz(3.128933285836239) q[2];
ry(1.1717892060325907) q[3];
rz(-2.8323694197867035) q[3];
ry(3.1240905081976273) q[4];
rz(2.295395639280342) q[4];
ry(-0.3339350406495418) q[5];
rz(-2.1361108951781445) q[5];
ry(2.3843195643274435) q[6];
rz(-1.6830411102774976) q[6];
ry(1.3351060488989974) q[7];
rz(0.5330374819064295) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.2406696237308315) q[0];
rz(-1.0036205037223882) q[0];
ry(-1.0248156954383925) q[1];
rz(1.4018547850300964) q[1];
ry(1.8942414503063922) q[2];
rz(0.7491703170571641) q[2];
ry(-2.936478851996105) q[3];
rz(-1.8328626111342297) q[3];
ry(2.751620875655604) q[4];
rz(1.2530909265911987) q[4];
ry(-2.9821971476159836) q[5];
rz(-1.7809453484048074) q[5];
ry(1.904308504200297) q[6];
rz(2.6906104864628944) q[6];
ry(-2.562314693126475) q[7];
rz(3.0286607420613234) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.01732369067652314) q[0];
rz(-0.12436794619387745) q[0];
ry(2.1852528606758157) q[1];
rz(-1.7466280354694335) q[1];
ry(-2.2553030957662603) q[2];
rz(-2.4336643693714306) q[2];
ry(-2.343895455017797) q[3];
rz(0.013471943896069627) q[3];
ry(-1.4528173493487508) q[4];
rz(1.1828859852123343) q[4];
ry(-3.1348978360058077) q[5];
rz(-2.017505720080906) q[5];
ry(2.752926619726147) q[6];
rz(0.3212457209432432) q[6];
ry(-1.6702985289040557) q[7];
rz(1.1814506161039424) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.6620005487008636) q[0];
rz(1.3930443724892356) q[0];
ry(1.265583392270457) q[1];
rz(-2.925233553230066) q[1];
ry(0.5184560610228512) q[2];
rz(1.7013892884181052) q[2];
ry(2.407027115706088) q[3];
rz(0.36273996534686687) q[3];
ry(2.89257240276435) q[4];
rz(-0.4611151064624034) q[4];
ry(-0.42429847847536184) q[5];
rz(2.696240461899107) q[5];
ry(-1.5065083372665706) q[6];
rz(-2.5613976452012803) q[6];
ry(2.338985218382182) q[7];
rz(-1.998111847627623) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.3576388528629986) q[0];
rz(-1.5426576772865495) q[0];
ry(-2.018191370244301) q[1];
rz(-1.974017428757296) q[1];
ry(1.746618623873712) q[2];
rz(-1.0678760060816659) q[2];
ry(-0.2389388935145087) q[3];
rz(3.029157255330868) q[3];
ry(-1.8263807750221213) q[4];
rz(2.779934541214964) q[4];
ry(-1.8371442768929542) q[5];
rz(-2.517967446105778) q[5];
ry(0.8865242229809525) q[6];
rz(0.34587210359607745) q[6];
ry(-0.6594691259013522) q[7];
rz(-1.843324865651848) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.5591183879504351) q[0];
rz(3.121031647037222) q[0];
ry(-2.047287770640307) q[1];
rz(3.0518631654820543) q[1];
ry(-0.7597764452886748) q[2];
rz(1.5270332952825287) q[2];
ry(0.3533776501289809) q[3];
rz(3.1090213370841258) q[3];
ry(2.5014198201471993) q[4];
rz(1.7199543497311502) q[4];
ry(1.6118949061827363) q[5];
rz(1.205026966610557) q[5];
ry(-1.855279782106738) q[6];
rz(-2.0908752428851427) q[6];
ry(-1.1322132093965918) q[7];
rz(1.4698751673739405) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.81726032265107) q[0];
rz(0.34810978105325674) q[0];
ry(-1.491213908891879) q[1];
rz(1.3768703478400823) q[1];
ry(-3.0479976020228405) q[2];
rz(-1.6513241340334703) q[2];
ry(2.908619570853493) q[3];
rz(-0.9785968679317836) q[3];
ry(-0.6905285161851367) q[4];
rz(1.93993556187643) q[4];
ry(-1.6198531404523369) q[5];
rz(-2.3062157766104336) q[5];
ry(-1.546838866152796) q[6];
rz(1.8488953707184186) q[6];
ry(-2.33468722172358) q[7];
rz(-3.035746251819928) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.0847211517356574) q[0];
rz(-1.6977175774525415) q[0];
ry(-2.3658717473938378) q[1];
rz(1.8373920105744552) q[1];
ry(0.6645222878810709) q[2];
rz(1.5253134530837675) q[2];
ry(-2.0043048056940487) q[3];
rz(-0.19758286137098127) q[3];
ry(-0.790290964722419) q[4];
rz(-0.31791544922490134) q[4];
ry(-0.9637341405927636) q[5];
rz(0.7322568829323836) q[5];
ry(1.3919663548610952) q[6];
rz(-2.9758104146936097) q[6];
ry(0.653534429326152) q[7];
rz(-1.1462861941331661) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.4967268318496751) q[0];
rz(1.6148222940851085) q[0];
ry(2.8850502095593127) q[1];
rz(-0.04598612913841659) q[1];
ry(2.5970642123396432) q[2];
rz(0.6383137313266606) q[2];
ry(1.3805909286656401) q[3];
rz(-1.5065002070981803) q[3];
ry(1.4870941561938467) q[4];
rz(-2.6616076396556108) q[4];
ry(0.8434758427910571) q[5];
rz(-0.22282937997227892) q[5];
ry(-0.5194994197077537) q[6];
rz(2.6125970523954534) q[6];
ry(0.6795782245222526) q[7];
rz(1.3793999092687192) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.094183090522293) q[0];
rz(-0.5677209566581618) q[0];
ry(1.9356434316671196) q[1];
rz(2.8843492995588735) q[1];
ry(-0.22008637811817167) q[2];
rz(-2.2643834534202822) q[2];
ry(-2.1440656555825215) q[3];
rz(-2.7965485931195246) q[3];
ry(-0.740089852426153) q[4];
rz(-1.754170295576837) q[4];
ry(-2.4452244316071776) q[5];
rz(-1.377663321234925) q[5];
ry(-2.0556204128304234) q[6];
rz(-1.5303186917176337) q[6];
ry(-2.0957717765766146) q[7];
rz(0.6922518513142687) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.223186875742365) q[0];
rz(0.4357953761628819) q[0];
ry(0.4457288774250297) q[1];
rz(-0.5774042122732626) q[1];
ry(-1.6482315291063818) q[2];
rz(-2.7721513970398677) q[2];
ry(1.630258525984958) q[3];
rz(-2.3861002460492937) q[3];
ry(2.9377562271341304) q[4];
rz(0.8941515138360776) q[4];
ry(-0.7094773506551206) q[5];
rz(2.843811983295339) q[5];
ry(-0.2057412052572536) q[6];
rz(1.7804011159450288) q[6];
ry(-1.3895859347648893) q[7];
rz(-0.5591238305472439) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.8901602401726747) q[0];
rz(-1.155358249793065) q[0];
ry(0.2520225271918487) q[1];
rz(0.8674141290890969) q[1];
ry(-0.40386263883391443) q[2];
rz(0.48238782958120735) q[2];
ry(-2.100294967774344) q[3];
rz(-2.7855368301188532) q[3];
ry(-1.9617959477184768) q[4];
rz(0.09566612799961362) q[4];
ry(1.2676997157042322) q[5];
rz(-2.27679872510241) q[5];
ry(1.7835921631294092) q[6];
rz(2.682670691945514) q[6];
ry(-2.353286209363062) q[7];
rz(2.91671544863717) q[7];