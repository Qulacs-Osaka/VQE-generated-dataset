OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.6770875250282318) q[0];
ry(-2.5258492367744774) q[1];
cx q[0],q[1];
ry(1.1219017812999263) q[0];
ry(-0.35645101711866956) q[1];
cx q[0],q[1];
ry(-1.160960363866433) q[2];
ry(0.2835724279291121) q[3];
cx q[2],q[3];
ry(-1.809733238675798) q[2];
ry(-1.1018251624434579) q[3];
cx q[2],q[3];
ry(1.4176338289014159) q[0];
ry(-1.3200673897517516) q[2];
cx q[0],q[2];
ry(-2.580367729680118) q[0];
ry(1.4516610003265258) q[2];
cx q[0],q[2];
ry(1.0463699816302512) q[1];
ry(0.019456962602391995) q[3];
cx q[1],q[3];
ry(0.663957268610849) q[1];
ry(0.8096900180099462) q[3];
cx q[1],q[3];
ry(1.5647625293369907) q[0];
ry(-0.44221289390234536) q[3];
cx q[0],q[3];
ry(-2.408114869897513) q[0];
ry(1.3961872149656729) q[3];
cx q[0],q[3];
ry(-1.2046048321436196) q[1];
ry(-1.5021240679705639) q[2];
cx q[1],q[2];
ry(2.1065614048988874) q[1];
ry(0.24784022514988374) q[2];
cx q[1],q[2];
ry(1.1985708068324312) q[0];
ry(0.8034276293553285) q[1];
cx q[0],q[1];
ry(-2.6330293467375534) q[0];
ry(2.1403242080022515) q[1];
cx q[0],q[1];
ry(2.516080451946367) q[2];
ry(0.8278976048377258) q[3];
cx q[2],q[3];
ry(-0.17152460026369265) q[2];
ry(2.217453834651877) q[3];
cx q[2],q[3];
ry(0.8520700054349284) q[0];
ry(-2.223370758012636) q[2];
cx q[0],q[2];
ry(0.753358480251063) q[0];
ry(-2.2862331831143776) q[2];
cx q[0],q[2];
ry(2.6292100960814415) q[1];
ry(-2.633528237238666) q[3];
cx q[1],q[3];
ry(-0.5289255387313941) q[1];
ry(-0.4567414527973517) q[3];
cx q[1],q[3];
ry(-0.18269994926317512) q[0];
ry(2.896138245233947) q[3];
cx q[0],q[3];
ry(-1.1649187512007142) q[0];
ry(2.701153655384447) q[3];
cx q[0],q[3];
ry(-2.1222760542282044) q[1];
ry(-3.0589159044423524) q[2];
cx q[1],q[2];
ry(-2.043694438219437) q[1];
ry(3.096812818441637) q[2];
cx q[1],q[2];
ry(2.8279626406362777) q[0];
ry(-0.7267389712839822) q[1];
cx q[0],q[1];
ry(2.994023095884832) q[0];
ry(-0.3762312304338407) q[1];
cx q[0],q[1];
ry(2.835720785316826) q[2];
ry(2.3557527979481545) q[3];
cx q[2],q[3];
ry(-0.4504896338390551) q[2];
ry(-2.20083645271612) q[3];
cx q[2],q[3];
ry(-1.2649509103133392) q[0];
ry(2.3043090439710605) q[2];
cx q[0],q[2];
ry(1.9154516806869264) q[0];
ry(0.6121430934107348) q[2];
cx q[0],q[2];
ry(-1.0059822036950021) q[1];
ry(-2.2363274744189585) q[3];
cx q[1],q[3];
ry(2.7989928372758763) q[1];
ry(2.7397207146184144) q[3];
cx q[1],q[3];
ry(-3.0400427553453957) q[0];
ry(-0.5538091233620118) q[3];
cx q[0],q[3];
ry(1.616198737044356) q[0];
ry(3.0016174907991386) q[3];
cx q[0],q[3];
ry(-0.6901977080121489) q[1];
ry(-1.5837222143341698) q[2];
cx q[1],q[2];
ry(-1.6412540029893998) q[1];
ry(1.5336596761972956) q[2];
cx q[1],q[2];
ry(0.9249790001831348) q[0];
ry(-2.113871820272899) q[1];
cx q[0],q[1];
ry(0.6345167712318246) q[0];
ry(0.391362406809386) q[1];
cx q[0],q[1];
ry(1.2312742325257968) q[2];
ry(2.7307765673434354) q[3];
cx q[2],q[3];
ry(-1.0440405230305734) q[2];
ry(1.7500127206707317) q[3];
cx q[2],q[3];
ry(1.3576026710646705) q[0];
ry(-1.1559228293614312) q[2];
cx q[0],q[2];
ry(-1.4903777421441147) q[0];
ry(-3.124556401648096) q[2];
cx q[0],q[2];
ry(2.6939221585750204) q[1];
ry(1.749954077970006) q[3];
cx q[1],q[3];
ry(-1.7211800808983562) q[1];
ry(1.8770502384560568) q[3];
cx q[1],q[3];
ry(-0.23876606429867983) q[0];
ry(-1.3637250715496056) q[3];
cx q[0],q[3];
ry(0.9229319968649461) q[0];
ry(-2.6864578707088462) q[3];
cx q[0],q[3];
ry(-1.2640925998971768) q[1];
ry(1.6098313365505157) q[2];
cx q[1],q[2];
ry(-1.8806610967076562) q[1];
ry(0.23082435498331666) q[2];
cx q[1],q[2];
ry(0.9874818984790004) q[0];
ry(-2.2191770962445085) q[1];
cx q[0],q[1];
ry(1.3808912982426174) q[0];
ry(0.5000280428929481) q[1];
cx q[0],q[1];
ry(-0.02033830171097861) q[2];
ry(0.748825852611664) q[3];
cx q[2],q[3];
ry(0.297010223182823) q[2];
ry(2.2359837858851694) q[3];
cx q[2],q[3];
ry(-1.1364785838215328) q[0];
ry(-0.5413480048686026) q[2];
cx q[0],q[2];
ry(0.257973217551566) q[0];
ry(0.05916971277317451) q[2];
cx q[0],q[2];
ry(3.1289411975977033) q[1];
ry(3.0472917191511226) q[3];
cx q[1],q[3];
ry(0.5993538419759797) q[1];
ry(1.5655654467046043) q[3];
cx q[1],q[3];
ry(0.9286672798296989) q[0];
ry(-2.2537160128144484) q[3];
cx q[0],q[3];
ry(3.0143151018434913) q[0];
ry(0.20801876733504712) q[3];
cx q[0],q[3];
ry(2.9884015249099383) q[1];
ry(-1.4901319373837183) q[2];
cx q[1],q[2];
ry(-0.36705459686340897) q[1];
ry(1.1629089405813862) q[2];
cx q[1],q[2];
ry(2.5844855164953446) q[0];
ry(-3.108078877231786) q[1];
cx q[0],q[1];
ry(0.4779870054743283) q[0];
ry(2.0299540631630686) q[1];
cx q[0],q[1];
ry(-1.3918265471255695) q[2];
ry(2.523835790882297) q[3];
cx q[2],q[3];
ry(-1.7256546357152152) q[2];
ry(-0.326724789939477) q[3];
cx q[2],q[3];
ry(2.832011596708781) q[0];
ry(0.5740595099056218) q[2];
cx q[0],q[2];
ry(2.162478589098016) q[0];
ry(0.2190016869685838) q[2];
cx q[0],q[2];
ry(0.9160935215325132) q[1];
ry(2.552698992995385) q[3];
cx q[1],q[3];
ry(2.8703549063889326) q[1];
ry(-0.46149824157114483) q[3];
cx q[1],q[3];
ry(2.358841760309921) q[0];
ry(-0.4583999439442623) q[3];
cx q[0],q[3];
ry(0.9406411336662419) q[0];
ry(-1.4303064284267482) q[3];
cx q[0],q[3];
ry(-2.634589169508685) q[1];
ry(3.064864555319417) q[2];
cx q[1],q[2];
ry(2.087230499055255) q[1];
ry(-1.21319557264842) q[2];
cx q[1],q[2];
ry(2.051495401018169) q[0];
ry(-1.3554808425030602) q[1];
cx q[0],q[1];
ry(0.7797829129243036) q[0];
ry(1.092908016088979) q[1];
cx q[0],q[1];
ry(0.40843917320929896) q[2];
ry(-0.7780256665353171) q[3];
cx q[2],q[3];
ry(-2.937269753253337) q[2];
ry(1.970517528251845) q[3];
cx q[2],q[3];
ry(2.443733040793349) q[0];
ry(0.25901605283178025) q[2];
cx q[0],q[2];
ry(-1.6540375616533858) q[0];
ry(0.395253046143484) q[2];
cx q[0],q[2];
ry(-0.6776955511695313) q[1];
ry(0.43802886121456375) q[3];
cx q[1],q[3];
ry(1.6565221370264842) q[1];
ry(2.5220032761328874) q[3];
cx q[1],q[3];
ry(0.44367902791189096) q[0];
ry(-1.3404290496512044) q[3];
cx q[0],q[3];
ry(-0.4498601703698091) q[0];
ry(-2.707504459637953) q[3];
cx q[0],q[3];
ry(1.2906043823855862) q[1];
ry(2.4365609990983073) q[2];
cx q[1],q[2];
ry(-1.1962023673291444) q[1];
ry(2.9870770237368016) q[2];
cx q[1],q[2];
ry(0.3206015280133636) q[0];
ry(1.4701904272278759) q[1];
cx q[0],q[1];
ry(-1.2768127749063067) q[0];
ry(-0.9957112518302553) q[1];
cx q[0],q[1];
ry(-1.9761910266482288) q[2];
ry(-2.0907628347085576) q[3];
cx q[2],q[3];
ry(-2.9159277135583386) q[2];
ry(-2.0700872454051944) q[3];
cx q[2],q[3];
ry(0.5744694776934542) q[0];
ry(1.24865517435153) q[2];
cx q[0],q[2];
ry(1.0173410822430968) q[0];
ry(-2.250999909424778) q[2];
cx q[0],q[2];
ry(-1.7959122794999773) q[1];
ry(0.07743378316554025) q[3];
cx q[1],q[3];
ry(-2.6449465088352078) q[1];
ry(-2.3372128979348212) q[3];
cx q[1],q[3];
ry(2.8599618968172518) q[0];
ry(0.8635332425113793) q[3];
cx q[0],q[3];
ry(-2.087433665626982) q[0];
ry(0.42559003217792857) q[3];
cx q[0],q[3];
ry(-2.428959875431337) q[1];
ry(2.974717287629828) q[2];
cx q[1],q[2];
ry(-1.5489962259555092) q[1];
ry(-2.6575445588944357) q[2];
cx q[1],q[2];
ry(-0.7369467415777828) q[0];
ry(-0.5993615690564447) q[1];
cx q[0],q[1];
ry(-2.456975895988574) q[0];
ry(0.46248586707220873) q[1];
cx q[0],q[1];
ry(3.0810351113431675) q[2];
ry(0.6166006618596578) q[3];
cx q[2],q[3];
ry(-0.3156112697393227) q[2];
ry(2.908720969630873) q[3];
cx q[2],q[3];
ry(0.9230247529844048) q[0];
ry(-1.4786021912503102) q[2];
cx q[0],q[2];
ry(-0.8590276433237789) q[0];
ry(-1.3983504855377387) q[2];
cx q[0],q[2];
ry(2.341832515994632) q[1];
ry(-1.0039009918091084) q[3];
cx q[1],q[3];
ry(-1.0127689575236654) q[1];
ry(-0.8949376534391065) q[3];
cx q[1],q[3];
ry(-0.9525684018762156) q[0];
ry(0.5240817848354435) q[3];
cx q[0],q[3];
ry(1.8686863142601076) q[0];
ry(-2.7370113452766205) q[3];
cx q[0],q[3];
ry(2.1756761243994585) q[1];
ry(0.3455371235037328) q[2];
cx q[1],q[2];
ry(-0.9026164134028987) q[1];
ry(1.810980627251177) q[2];
cx q[1],q[2];
ry(2.5940646874707745) q[0];
ry(-2.5814626726821857) q[1];
cx q[0],q[1];
ry(-0.01743659937632933) q[0];
ry(2.466274530087267) q[1];
cx q[0],q[1];
ry(2.244114514276637) q[2];
ry(1.8718667024652573) q[3];
cx q[2],q[3];
ry(1.5892986055810132) q[2];
ry(-0.5268499414950893) q[3];
cx q[2],q[3];
ry(2.842360979425925) q[0];
ry(-0.8633487367428413) q[2];
cx q[0],q[2];
ry(0.13917767784141422) q[0];
ry(-3.0343013598938477) q[2];
cx q[0],q[2];
ry(-0.2883912519295765) q[1];
ry(0.4404123900496772) q[3];
cx q[1],q[3];
ry(1.1258601578601533) q[1];
ry(-1.140374931157231) q[3];
cx q[1],q[3];
ry(0.6695702139594064) q[0];
ry(2.2219931493114364) q[3];
cx q[0],q[3];
ry(2.0105056486278574) q[0];
ry(2.623376397553059) q[3];
cx q[0],q[3];
ry(-2.224793943409835) q[1];
ry(-2.7375156170543713) q[2];
cx q[1],q[2];
ry(0.7557256241628054) q[1];
ry(2.2920963449431815) q[2];
cx q[1],q[2];
ry(1.8669913267734521) q[0];
ry(2.862556341120633) q[1];
ry(-1.1102206918792135) q[2];
ry(-1.2388548528695313) q[3];