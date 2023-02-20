OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.8176215489482281) q[0];
rz(0.07441977732489402) q[0];
ry(-0.9434135727522923) q[1];
rz(-0.7196141462132273) q[1];
ry(-2.663478067625514) q[2];
rz(-1.6641262526318725) q[2];
ry(3.0843561732891853) q[3];
rz(2.7740993291828393) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.128408151588162) q[0];
rz(2.2949639516329183) q[0];
ry(-0.8080807042622066) q[1];
rz(-2.8999630021739153) q[1];
ry(-2.5014256155907795) q[2];
rz(1.4442860652205525) q[2];
ry(-2.101986676875085) q[3];
rz(1.2654748854494935) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.5093371322070217) q[0];
rz(1.022702072963476) q[0];
ry(-0.5622887173330322) q[1];
rz(-1.9583294241070914) q[1];
ry(-0.7348714122379021) q[2];
rz(1.386425468711748) q[2];
ry(-2.1082919126495514) q[3];
rz(-1.311711794146168) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.157741674014637) q[0];
rz(-2.0183198989405087) q[0];
ry(0.912794661275321) q[1];
rz(-0.22913831996592915) q[1];
ry(0.717911789087709) q[2];
rz(-0.3921929520939765) q[2];
ry(-0.0547114176053796) q[3];
rz(-0.5616827124140588) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.6376660034847865) q[0];
rz(2.7129632803941033) q[0];
ry(-2.0085993357731793) q[1];
rz(-2.6592179545976995) q[1];
ry(0.3145409968323514) q[2];
rz(2.5925314383281597) q[2];
ry(0.3667041576880141) q[3];
rz(-3.110139080108398) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.6510738608652087) q[0];
rz(1.015965368567924) q[0];
ry(-0.9801527417266946) q[1];
rz(-0.4283800466298404) q[1];
ry(2.8620201754481154) q[2];
rz(-0.1379651397155479) q[2];
ry(3.1403054107290864) q[3];
rz(-0.21293698987226162) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.685333841933871) q[0];
rz(-2.9499513697307815) q[0];
ry(0.04221118360551479) q[1];
rz(1.03249463511974) q[1];
ry(2.473586345101779) q[2];
rz(-0.5734470652415785) q[2];
ry(1.9492796583938354) q[3];
rz(-0.9537758632617787) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2288890361612017) q[0];
rz(-0.9169771160418759) q[0];
ry(0.014329889863141433) q[1];
rz(2.9782343192896423) q[1];
ry(1.775227898011803) q[2];
rz(-1.6084421261553281) q[2];
ry(1.4379870862684268) q[3];
rz(-1.544265008405107) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.073724616616337) q[0];
rz(-1.2703885811575386) q[0];
ry(-0.18032335571122152) q[1];
rz(-1.6723974119733565) q[1];
ry(-2.467616425957587) q[2];
rz(0.18203289308084525) q[2];
ry(2.643910093406597) q[3];
rz(-1.1071345133682948) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.023379315360554) q[0];
rz(-2.173461068003503) q[0];
ry(-2.216281165910078) q[1];
rz(-1.2852143327437293) q[1];
ry(-2.2281856174744044) q[2];
rz(0.9161944076498686) q[2];
ry(2.384895090348565) q[3];
rz(0.6994259884167775) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.49331016407095696) q[0];
rz(3.1263091360062822) q[0];
ry(0.31733621984410654) q[1];
rz(-0.7587030731064877) q[1];
ry(0.9159144864015447) q[2];
rz(2.7793752875847915) q[2];
ry(0.9160258712973732) q[3];
rz(-2.6712364694567134) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.0079574440374985) q[0];
rz(-0.21672942699374698) q[0];
ry(-1.3115726493487427) q[1];
rz(0.09074429001676035) q[1];
ry(-1.9396365552584365) q[2];
rz(-2.3991494453149493) q[2];
ry(0.2376203265918381) q[3];
rz(-2.8614170689700957) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.1197240147824061) q[0];
rz(-0.3931253831440132) q[0];
ry(2.050817808381608) q[1];
rz(-1.420681931604867) q[1];
ry(0.5990830716352977) q[2];
rz(1.18359027221452) q[2];
ry(0.263294384557752) q[3];
rz(1.6675842891676569) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.5256106837975194) q[0];
rz(-1.9311564751393204) q[0];
ry(2.293752533525729) q[1];
rz(-0.582915560588705) q[1];
ry(0.10131819274558218) q[2];
rz(0.989566443084003) q[2];
ry(-1.888211581613705) q[3];
rz(0.7213007826078017) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8532641982655478) q[0];
rz(-2.1695627913676496) q[0];
ry(-2.5257054190154906) q[1];
rz(-2.1196539047141147) q[1];
ry(-0.057215187217184216) q[2];
rz(2.84863823861953) q[2];
ry(3.053047402730002) q[3];
rz(0.7676633949207475) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.436504958155063) q[0];
rz(2.032813806759135) q[0];
ry(-1.0649173635094513) q[1];
rz(-2.6026367326346063) q[1];
ry(1.0465508462100697) q[2];
rz(-2.595043764267488) q[2];
ry(-1.0888102791729128) q[3];
rz(2.291677757577569) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.010803051140749) q[0];
rz(-0.14779169477531084) q[0];
ry(-2.6301777132151742) q[1];
rz(1.0004445160280493) q[1];
ry(-1.1744455035936123) q[2];
rz(1.0507007883300457) q[2];
ry(2.1833189062126603) q[3];
rz(-2.4568297343725973) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.9977019665398217) q[0];
rz(0.061045869658200184) q[0];
ry(2.8836750022945874) q[1];
rz(0.726944280486248) q[1];
ry(-2.7961544723372405) q[2];
rz(-1.977451294795733) q[2];
ry(1.1815833149199315) q[3];
rz(2.5578178238307494) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.237172405839249) q[0];
rz(-2.747800627928562) q[0];
ry(-2.876088096737426) q[1];
rz(-0.7474696816901938) q[1];
ry(1.3962509085006705) q[2];
rz(1.918029498204846) q[2];
ry(0.9958096426351268) q[3];
rz(-1.1606738186287338) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.11475455948303258) q[0];
rz(2.8546433294648876) q[0];
ry(-1.2934436966902183) q[1];
rz(-2.381383394426981) q[1];
ry(3.0715153493071847) q[2];
rz(-1.3975172657938018) q[2];
ry(0.29567491865609075) q[3];
rz(-2.685708535599583) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.3631237708818387) q[0];
rz(0.8871771090838676) q[0];
ry(1.2129044612604236) q[1];
rz(-1.2694964796863106) q[1];
ry(-0.4243528335806554) q[2];
rz(0.5812956647758242) q[2];
ry(1.2278076552093082) q[3];
rz(3.024131275226543) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.7566736469574455) q[0];
rz(-0.964527663200278) q[0];
ry(2.371384239327375) q[1];
rz(1.1506541275658706) q[1];
ry(3.047093793095719) q[2];
rz(-0.38175819343403644) q[2];
ry(2.2559132121224357) q[3];
rz(0.6931714589036083) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.3331670588037972) q[0];
rz(0.26693268165576234) q[0];
ry(-2.628743460951263) q[1];
rz(-0.004825979159710884) q[1];
ry(-2.5245910273725207) q[2];
rz(2.0416627885257426) q[2];
ry(-0.8667379134667628) q[3];
rz(-0.0054338984731385764) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.4679690096235705) q[0];
rz(2.3885232962530596) q[0];
ry(3.115352882601719) q[1];
rz(0.659345806928336) q[1];
ry(1.0300449870442652) q[2];
rz(-2.2576009632769334) q[2];
ry(-2.3609585872801753) q[3];
rz(0.9149630564829825) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6842391684893063) q[0];
rz(-2.575213862431823) q[0];
ry(0.9751129823587372) q[1];
rz(-1.9201114953579634) q[1];
ry(-0.7897981532639758) q[2];
rz(0.7303666533130674) q[2];
ry(0.495887580830388) q[3];
rz(-2.007103589186608) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.9111437509126388) q[0];
rz(-1.5259068652098164) q[0];
ry(-2.239812715780368) q[1];
rz(0.3238845758403875) q[1];
ry(-2.436617577933592) q[2];
rz(-2.177539787818035) q[2];
ry(1.536410922533775) q[3];
rz(1.4087478294996318) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.5976540069668481) q[0];
rz(-1.000593415466289) q[0];
ry(1.025659792900672) q[1];
rz(-0.38856042981529504) q[1];
ry(-1.0069871636892156) q[2];
rz(-2.6172148130110147) q[2];
ry(-1.6177498447263847) q[3];
rz(1.2042503212149045) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2974335445560188) q[0];
rz(0.5995281406362483) q[0];
ry(0.7099748128576371) q[1];
rz(-2.0493466134732476) q[1];
ry(2.454192330782233) q[2];
rz(-2.646682121693232) q[2];
ry(0.587264073777793) q[3];
rz(1.4620033220174011) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.604922588604166) q[0];
rz(-3.1129034507081075) q[0];
ry(0.4282113946539619) q[1];
rz(-2.588114634479808) q[1];
ry(1.1690944729032324) q[2];
rz(2.2932366965183113) q[2];
ry(1.6068479897559795) q[3];
rz(-0.9253714257496062) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.8644141187961054) q[0];
rz(-2.344891408687528) q[0];
ry(1.986424436772233) q[1];
rz(1.3923635365530638) q[1];
ry(-1.9866056906211238) q[2];
rz(2.5771027952490337) q[2];
ry(1.3430250727508959) q[3];
rz(-3.076017860169031) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.600295381279398) q[0];
rz(-2.830394952439423) q[0];
ry(-1.8492750994083806) q[1];
rz(-0.24039435485649288) q[1];
ry(1.0007470420618994) q[2];
rz(-0.8144461332922862) q[2];
ry(0.5553264490501332) q[3];
rz(-1.0374106865251616) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.5425419203543322) q[0];
rz(-0.899778732099758) q[0];
ry(-1.845931463014923) q[1];
rz(-1.6084427909146937) q[1];
ry(2.7767204518505255) q[2];
rz(-1.1737861380651866) q[2];
ry(0.028124295986792443) q[3];
rz(2.137698468568623) q[3];