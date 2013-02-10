
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
  "use strict";
  
  var sceneWidth = 1000;
  var sceneHeight = 500;
  var borderFactor = 6 / 5;
  
  var itemCount = 5000;
  var radius = 10;
  
  phet.tests.sceneDirtyRegions = function( main, moveCount ) {
    var scene = new scenery.Scene( main );
    var root = scene.root;
    
    var background = new scenery.Path();
    background.setShape( scenery.Shape.rectangle( -sceneWidth / 2 * borderFactor, -sceneHeight / 2 * borderFactor, sceneWidth * borderFactor, sceneHeight * borderFactor ) );
    background.setFill( '#333333' );
    background.setStroke( '#000000' );
    root.addChild( background );
    
    var nodes = new scenery.Node();
    root.addChild( nodes );
    
    for ( var i = 0; i < itemCount; i++ ) {
      var node = new scenery.Path();
      
      // regular polygon
      node.setShape( scenery.Shape.regularPolygon( 6, radius ) );
      
      var xFactor = Math.random();
      node.setTranslation( ( xFactor - 0.5 ) * sceneWidth, ( Math.random() - 0.5 ) * sceneHeight );
      
      // TODO: better way of specifying initial parameters here would be ideal
      node.setFill( phet.tests.themeColor( 0.5, xFactor ) );
      node.setStroke( '#000000' );
      
      nodes.addChild( node );
    }
    
    // center the root
    root.translate( main.width() / 2, main.height() / 2 );
    
    // return step function
    return function( timeElapsed ) {
      if ( moveCount == 0 ) {
        for ( var i = 0; i < itemCount; i++ ) {
          nodes.children[i].translate( ( Math.random() - 0.5 ) * 50, ( Math.random() - 0.5 ) * 50 );
        }
      } else {
        for ( var j = 0; j < moveCount; j++ ) {
        // tweak a random node
          var node = nodes.children[_.random( 0, nodes.children.length - 1)];
          node.translate( ( Math.random() - 0.5 ) * 50, ( Math.random() - 0.5 ) * 50 );
        }
      }
      
      scene.updateScene();
    }
  };
  
  phet.tests.easelDirtyRegions = function( main, moveCount ) {
    var canvas = document.createElement( 'canvas' );
    canvas.id = 'easel-canvas';
    canvas.width = main.width();
    canvas.height = main.height();
    main.append( canvas );

    var stage = new createjs.Stage( canvas );
    
    var background = new createjs.Shape();
    background.graphics.beginFill( '#333333').beginStroke( '#000000' ).drawRect(  -sceneWidth / 2 * borderFactor, -sceneHeight / 2 * borderFactor, sceneWidth * borderFactor, sceneHeight * borderFactor );
    stage.addChild( background );
    
    var nodes = new createjs.Container();
    stage.addChild( nodes );
    
    for ( var i = 0; i < itemCount; i++ ) {
      var shape = new createjs.Shape();
      
      var xFactor = Math.random();
      
      shape.graphics.beginFill( phet.tests.themeColor( 0.5, xFactor ) ).beginStroke( '#000000' ).drawPolyStar( 0, 0, radius, 6, 0, 0 );
      
      shape.x = ( xFactor - 0.5 ) * sceneWidth;
      shape.y = ( Math.random() - 0.5 ) * sceneHeight;
      
      nodes.addChild( shape );
    }
    
    stage.x = main.width() / 2;
    stage.y = main.height() / 2;
    
    // return step function
    return function( timeElapsed ) {
      if ( moveCount == 0 ) {
        for ( var i = 0; i < itemCount; i++ ) {
          var shape = nodes.children[i];
          shape.x += ( Math.random() - 0.5 ) * 50;
          shape.y += ( Math.random() - 0.5 ) * 50;
        }
      } else {
        for ( var j = 0; j < moveCount; j++ ) {
          var shape = nodes.children[_.random( 0, nodes.children.length - 1)];
          shape.x += ( Math.random() - 0.5 ) * 50;
          shape.y += ( Math.random() - 0.5 ) * 50;
        }
      }
      
      stage.update();
    }
  };  
  
})();

