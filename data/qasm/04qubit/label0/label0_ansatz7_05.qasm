OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.5899583871010554) q[0];
ry(0.5058829033799199) q[1];
cx q[0],q[1];
ry(0.01824800735510922) q[0];
ry(2.7473948005181676) q[1];
cx q[0],q[1];
ry(1.673433138861114) q[0];
ry(1.2335653443342383) q[2];
cx q[0],q[2];
ry(-0.2306579313931918) q[0];
ry(0.8748911075441004) q[2];
cx q[0],q[2];
ry(-2.544669170320442) q[0];
ry(-1.3577219830311344) q[3];
cx q[0],q[3];
ry(-0.7856993795082143) q[0];
ry(-0.2852231306464432) q[3];
cx q[0],q[3];
ry(-2.8201800295515116) q[1];
ry(0.7570334519191809) q[2];
cx q[1],q[2];
ry(2.1231969541004654) q[1];
ry(2.8284094180369577) q[2];
cx q[1],q[2];
ry(1.3958588851720588) q[1];
ry(-2.2054797610892463) q[3];
cx q[1],q[3];
ry(1.0612570501683545) q[1];
ry(-1.764750436864096) q[3];
cx q[1],q[3];
ry(-2.0125482020371024) q[2];
ry(-0.9887478280328672) q[3];
cx q[2],q[3];
ry(-0.3172295311578308) q[2];
ry(0.19114188692102424) q[3];
cx q[2],q[3];
ry(2.896652359424004) q[0];
ry(-1.5959234954822072) q[1];
cx q[0],q[1];
ry(0.2598920246809743) q[0];
ry(-0.36111062963416535) q[1];
cx q[0],q[1];
ry(2.0718220517524384) q[0];
ry(-2.3037706375965255) q[2];
cx q[0],q[2];
ry(0.37931801202221127) q[0];
ry(1.6317139722011897) q[2];
cx q[0],q[2];
ry(2.7699430570105332) q[0];
ry(-2.317423025065675) q[3];
cx q[0],q[3];
ry(2.891213720181848) q[0];
ry(1.8983639410207758) q[3];
cx q[0],q[3];
ry(0.9063526651346088) q[1];
ry(-0.7002926133364591) q[2];
cx q[1],q[2];
ry(1.2651767283911608) q[1];
ry(-1.3232253512778502) q[2];
cx q[1],q[2];
ry(2.2364996870148532) q[1];
ry(-1.2751847016978148) q[3];
cx q[1],q[3];
ry(-2.769745842301547) q[1];
ry(-3.1132519277449244) q[3];
cx q[1],q[3];
ry(0.8136726114851048) q[2];
ry(-1.1299220444840277) q[3];
cx q[2],q[3];
ry(-2.5067453132193713) q[2];
ry(-2.4974260330417355) q[3];
cx q[2],q[3];
ry(-2.165202450400861) q[0];
ry(0.4851108696178691) q[1];
cx q[0],q[1];
ry(-1.1205880546681977) q[0];
ry(2.635183436880239) q[1];
cx q[0],q[1];
ry(-2.1250698932109247) q[0];
ry(1.3311124798322365) q[2];
cx q[0],q[2];
ry(-2.529761846163884) q[0];
ry(-1.0030816848390298) q[2];
cx q[0],q[2];
ry(-0.7652670158925029) q[0];
ry(2.0794422451314194) q[3];
cx q[0],q[3];
ry(-1.0731905770120047) q[0];
ry(-1.4792954346294147) q[3];
cx q[0],q[3];
ry(-2.5282518788343302) q[1];
ry(0.4520624328330971) q[2];
cx q[1],q[2];
ry(-2.639196283639376) q[1];
ry(0.43891548392448154) q[2];
cx q[1],q[2];
ry(2.005162869699199) q[1];
ry(-2.388363542384063) q[3];
cx q[1],q[3];
ry(3.0742496667179604) q[1];
ry(2.326648175983367) q[3];
cx q[1],q[3];
ry(-2.337938192574928) q[2];
ry(-2.1206253738116194) q[3];
cx q[2],q[3];
ry(2.007459528665689) q[2];
ry(1.9732505303403758) q[3];
cx q[2],q[3];
ry(1.2249271077137658) q[0];
ry(3.1407150893176077) q[1];
cx q[0],q[1];
ry(2.7371317508414186) q[0];
ry(0.059032270944535455) q[1];
cx q[0],q[1];
ry(2.6827531261278676) q[0];
ry(0.23478124654460863) q[2];
cx q[0],q[2];
ry(-2.123219316394408) q[0];
ry(-0.5121339937769117) q[2];
cx q[0],q[2];
ry(3.0194405631351566) q[0];
ry(2.946006012760492) q[3];
cx q[0],q[3];
ry(-2.1464941963290345) q[0];
ry(-2.995751803835601) q[3];
cx q[0],q[3];
ry(1.0338547603963957) q[1];
ry(-2.071556972680173) q[2];
cx q[1],q[2];
ry(-2.0914401285466586) q[1];
ry(-2.6755636221063104) q[2];
cx q[1],q[2];
ry(-0.08665463121909783) q[1];
ry(-0.3435435784457308) q[3];
cx q[1],q[3];
ry(-2.5400156229598094) q[1];
ry(1.2526151179633054) q[3];
cx q[1],q[3];
ry(0.7652737284236384) q[2];
ry(0.8543353871337436) q[3];
cx q[2],q[3];
ry(-1.5721591070736964) q[2];
ry(-3.0560037241810147) q[3];
cx q[2],q[3];
ry(0.8024085873797382) q[0];
ry(-1.7302194395773172) q[1];
cx q[0],q[1];
ry(-0.28330654542719325) q[0];
ry(1.5323297021104532) q[1];
cx q[0],q[1];
ry(-1.380848220842084) q[0];
ry(2.039671255483615) q[2];
cx q[0],q[2];
ry(-1.9593653391785357) q[0];
ry(-0.683559823785572) q[2];
cx q[0],q[2];
ry(0.18917964450498814) q[0];
ry(0.13667129949022375) q[3];
cx q[0],q[3];
ry(-1.5755881272708083) q[0];
ry(0.9263057597846885) q[3];
cx q[0],q[3];
ry(-1.4468520460684022) q[1];
ry(-1.47745838520712) q[2];
cx q[1],q[2];
ry(1.641220476147029) q[1];
ry(0.4221614874655297) q[2];
cx q[1],q[2];
ry(0.2208644853411208) q[1];
ry(1.1491556202699593) q[3];
cx q[1],q[3];
ry(1.2567497570837898) q[1];
ry(1.978071601154448) q[3];
cx q[1],q[3];
ry(-2.9490811510058776) q[2];
ry(2.469072439406073) q[3];
cx q[2],q[3];
ry(-2.851344108818896) q[2];
ry(-2.29019301302242) q[3];
cx q[2],q[3];
ry(2.005032978537246) q[0];
ry(-2.23731391490777) q[1];
cx q[0],q[1];
ry(0.6458152514150842) q[0];
ry(-0.7919615791849743) q[1];
cx q[0],q[1];
ry(0.352529033875725) q[0];
ry(-1.0472209209294663) q[2];
cx q[0],q[2];
ry(1.5218470405078326) q[0];
ry(-2.112474697352611) q[2];
cx q[0],q[2];
ry(3.108451381741173) q[0];
ry(1.5276309325042332) q[3];
cx q[0],q[3];
ry(-2.7223717634271263) q[0];
ry(-0.9784020275954319) q[3];
cx q[0],q[3];
ry(-2.1395863151274366) q[1];
ry(2.053534953701621) q[2];
cx q[1],q[2];
ry(0.6238217572789255) q[1];
ry(-2.112418249503835) q[2];
cx q[1],q[2];
ry(2.2856332716902044) q[1];
ry(-1.7961828166098694) q[3];
cx q[1],q[3];
ry(2.2695751968398405) q[1];
ry(1.4364702388185342) q[3];
cx q[1],q[3];
ry(1.654221487756001) q[2];
ry(1.9174424194061228) q[3];
cx q[2],q[3];
ry(-1.775014790700852) q[2];
ry(2.976277947553058) q[3];
cx q[2],q[3];
ry(1.9931935016408515) q[0];
ry(1.360984527568609) q[1];
cx q[0],q[1];
ry(-0.3622815950921972) q[0];
ry(2.5268907054697785) q[1];
cx q[0],q[1];
ry(1.6849973517062296) q[0];
ry(1.947867824787565) q[2];
cx q[0],q[2];
ry(-1.355294547232582) q[0];
ry(-1.0927102708476981) q[2];
cx q[0],q[2];
ry(-1.3164744231828456) q[0];
ry(2.415307998744136) q[3];
cx q[0],q[3];
ry(0.6294024252191339) q[0];
ry(2.1551663545261506) q[3];
cx q[0],q[3];
ry(2.0099741596199197) q[1];
ry(2.0058913126492843) q[2];
cx q[1],q[2];
ry(-2.1342532073058567) q[1];
ry(0.2773507516547662) q[2];
cx q[1],q[2];
ry(-1.3796386848196045) q[1];
ry(-2.777438437754946) q[3];
cx q[1],q[3];
ry(2.96976419249346) q[1];
ry(1.653355428948402) q[3];
cx q[1],q[3];
ry(2.5031433523475877) q[2];
ry(2.7282569683930435) q[3];
cx q[2],q[3];
ry(0.19861304241670777) q[2];
ry(2.4013863776367543) q[3];
cx q[2],q[3];
ry(0.25734330577834896) q[0];
ry(2.5891113103264525) q[1];
cx q[0],q[1];
ry(-0.09855766183891562) q[0];
ry(2.1808758635558836) q[1];
cx q[0],q[1];
ry(-1.4301469917297558) q[0];
ry(0.21882959337461824) q[2];
cx q[0],q[2];
ry(-2.112218838545938) q[0];
ry(2.4717164341017956) q[2];
cx q[0],q[2];
ry(-2.309063949345558) q[0];
ry(0.688624267289176) q[3];
cx q[0],q[3];
ry(-1.691438413115057) q[0];
ry(-2.0512304294627564) q[3];
cx q[0],q[3];
ry(-2.815682921709272) q[1];
ry(-2.8572367068709705) q[2];
cx q[1],q[2];
ry(-2.6871713563241526) q[1];
ry(0.1343042387466422) q[2];
cx q[1],q[2];
ry(-1.6689564735824927) q[1];
ry(2.6723437212754426) q[3];
cx q[1],q[3];
ry(0.05524596770053864) q[1];
ry(0.011571993124653483) q[3];
cx q[1],q[3];
ry(0.9733784793862821) q[2];
ry(-0.997471916185007) q[3];
cx q[2],q[3];
ry(1.2505981699496855) q[2];
ry(1.7212138844916849) q[3];
cx q[2],q[3];
ry(-1.3742671583899555) q[0];
ry(-1.7905254242997692) q[1];
ry(-1.30003758607947) q[2];
ry(0.6332681465165563) q[3];