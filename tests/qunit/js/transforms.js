
(function(){
  'use strict';
  
  module( 'Scenery: Transforms' );
  
  var epsilon = 0.000000001;
  
  test( 'Points', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    a.addChild( b );
    a.x = 10;
    b.y = 10;
    
    ok( dot( 5, 15 ).equalsEpsilon( b.localToParentPoint( dot( 5, 5 ) ), epsilon ), 'localToParentPoint on child' );
    ok( dot( 15, 5 ).equalsEpsilon( a.localToParentPoint( dot( 5, 5 ) ), epsilon ), 'localToParentPoint on root' );
    
    ok( dot( 5, -5 ).equalsEpsilon( b.parentToLocalPoint( dot( 5, 5 ) ), epsilon ), 'parentToLocalPoint on child' );
    ok( dot( -5, 5 ).equalsEpsilon( a.parentToLocalPoint( dot( 5, 5 ) ), epsilon ), 'parentToLocalPoint on root' );
    
    ok( dot( 15, 15 ).equalsEpsilon( b.localToGlobalPoint( dot( 5, 5 ) ), epsilon ), 'localToGlobalPoint on child' );
    ok( dot( 15, 5 ).equalsEpsilon( a.localToGlobalPoint( dot( 5, 5 ) ), epsilon ), 'localToGlobalPoint on root (same as localToparent)' );
    
    ok( dot( -5, -5 ).equalsEpsilon( b.globalToLocalPoint( dot( 5, 5 ) ), epsilon ), 'globalToLocalPoint on child' );
    ok( dot( -5, 5 ).equalsEpsilon( a.globalToLocalPoint( dot( 5, 5 ) ), epsilon ), 'globalToLocalPoint on root (same as localToparent)' );
    
    ok( dot( 15, 5 ).equalsEpsilon( b.parentToGlobalPoint( dot( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on child' );
    ok( dot( 5, 5 ).equalsEpsilon( a.parentToGlobalPoint( dot( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on root' );
  } );
  
})();
