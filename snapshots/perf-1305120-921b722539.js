
(function(){
  
  var textNodeInstances = {
    name: 'Text node instances',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      this.i = 0;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      var text = new scenery.Text( 'A', { font: '16px sans-serif' } );
      for ( var i = 0; i < 10000; i++ ) {
        scene.addChild( new scenery.Node( {
          children: [text],
          x: i % 759,
          y: ( i * 172 ) % 973
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      this.scene.rotate( 0.01 );
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var textPathInstances = {
    name: 'Text path instances',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      this.i = 0;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      var text = new scenery.Path( {
        shape: new kite.Shape( 'M108 385.875 L82.6875 385.875 L82.6875 224.5781 Q63.7031 242.5781 31.9219 255.0938 L31.9219 230.625 Q75.6562 209.6719 91.5469 178.7344 L108 178.7344 L108 385.875 Z' ),
        fill: '#000',
        scale: 0.1
      } );
      for ( var i = 0; i < 10000; i++ ) {
        scene.addChild( new scenery.Node( {
          children: [text],
          x: i % 759,
          y: ( i * 172 ) % 973
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      this.scene.rotate( 0.01 );
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };

  var textBoundsChanges = {
    name: 'Text bounds changes',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      this.i = 0;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      this.text = new scenery.Text( 'm', {
        x: 10,
        centerY: this.height / 2,
        font: '40px Arial, sans-serif'
      } );
      scene.addChild( this.text );
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      var that = this;
      _.times( 20, function() {
        that.text.text = that.i++;
      } );
      // this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var fuzzRecordAddRemoveRender = {
    name: 'Fuzz record (add/remove/renderer)',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      this.i = 0;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      var scene = this.scene;
      scene.renderer = null;
      _.each( scene.children.slice( 0 ), function( child ) { scene.removeChild( child ); } );
      
      var node3 = new scenery.Node( {} );
      var node4 = new scenery.Node( {} );
      var node5 = new scenery.Node( {} );
      var node6 = new scenery.Node( {} );
      var node7 = new scenery.Node( {} );
      var path8 = new scenery.Path( {} );
      var path9 = new scenery.Path( {} );
      var path10 = new scenery.Path( {} );
      var path11 = new scenery.Path( {} );
      var path12 = new scenery.Path( {} );
      var path13 = new scenery.Path( {} );
      var path14 = new scenery.Path( {} );
      var path15 = new scenery.Path( {} );
      var path16 = new scenery.Path( {} );
      var path17 = new scenery.Path( {} );
      path12.renderer = null;
      path16.insertChild( 0, node5 );
      scene.renderer = 'canvas';
      scene.renderer = 'svg';
      path13.insertChild( 0, node6 );
      path13.insertChild( 1, path17 );
      node5.renderer = null;
      scene.insertChild( 0, path12 );
      node7.insertChild( 0, path15 );
      path13.renderer = null;
      path12.renderer = null;
      path12.insertChild( 0, node4 );
      path13.removeChild( node6 );
      path9.renderer = 'svg';
      node7.insertChild( 0, node5 );
      path13.removeChild( path17 );
      path10.renderer = null;
      path12.renderer = 'canvas';
      node3.insertChild( 0, path11 );
      path15.insertChild( 0, node5 );
      node6.insertChild( 0, path16 );
      path16.renderer = null;
      path9.renderer = 'canvas';
      node7.renderer = 'canvas';
      node7.removeChild( path15 );
      node6.insertChild( 0, path8 );
      path15.renderer = null;
      path11.renderer = null;
      path16.renderer = null;
      node7.removeChild( node5 );
      path12.insertChild( 0, path13 );
      node4.renderer = null;
      path9.renderer = null;
      node6.removeChild( path8 );
      path11.insertChild( 0, path9 );
      path12.renderer = null;
      path8.insertChild( 0, path16 );
      path10.insertChild( 0, node4 );
      path10.renderer = 'svg';
      path15.removeChild( node5 );
      path11.removeChild( path9 );
      path14.renderer = null;
      path8.renderer = null;
      node4.renderer = null;
      path16.renderer = null;
      path10.insertChild( 0, node7 );
      path8.insertChild( 0, node3 );
      scene.removeChild( path12 );
      path13.renderer = 'canvas';
      path11.insertChild( 0, path10 );
      path11.insertChild( 1, path15 );
      node5.renderer = null;
      path15.renderer = 'svg';
      path15.renderer = 'canvas';
      path12.insertChild( 1, path10 );
      path15.renderer = null;
      path10.insertChild( 1, path13 );
      node3.insertChild( 0, scene );
      path14.insertChild( 0, node3 );
      node6.insertChild( 1, path11 );
      path9.renderer = null;
      path16.renderer = 'canvas';
      scene.insertChild( 0, path10 );
      node5.renderer = null;
      node3.renderer = null;
      path12.removeChild( path13 );
      path12.insertChild( 2, node3 );
      node3.renderer = null;
      node4.insertChild( 0, node5 );
      path13.insertChild( 0, node5 );
      path13.insertChild( 0, path15 );
      path15.renderer = 'svg';
      path11.renderer = null;
      path15.renderer = 'canvas';
      path11.removeChild( path10 );
      path13.renderer = null;
      path10.renderer = null;
      scene.renderer = null;
      path13.renderer = null;
      node7.renderer = null;
      path13.renderer = null;
      node5.renderer = null;
      path12.renderer = null;
      node4.removeChild( node5 );
      path16.renderer = 'svg';
      path8.insertChild( 0, path17 );
      path9.insertChild( 0, path15 );
      node3.renderer = null;
      node6.insertChild( 0, path14 );
      path16.insertChild( 0, path12 );
      path12.insertChild( 3, path14 );
      node5.renderer = 'canvas';
      path10.renderer = 'canvas';
      path11.removeChild( path15 );
      node4.renderer = 'svg';
      path11.renderer = null;
      path8.renderer = 'canvas';
      node7.insertChild( 0, node5 );
      path14.renderer = null;
      path8.removeChild( node3 );
      path16.removeChild( path12 );
      node3.removeChild( scene );
      node4.renderer = null;
      node6.renderer = null;
      path12.renderer = 'svg';
      path16.renderer = 'svg';
      path14.removeChild( node3 );
      path16.renderer = 'canvas';
      node6.removeChild( path11 );
      path16.removeChild( node5 );
      path8.renderer = 'canvas';
      scene.renderer = null;
      path9.removeChild( path15 );
      node4.renderer = null;
      path17.insertChild( 0, scene );
      node6.removeChild( path16 );
      path14.renderer = null;
      path8.removeChild( path17 );
      path12.removeChild( node4 );
      path14.renderer = 'svg';
      scene.removeChild( path10 );
      node3.insertChild( 1, node5 );
      path12.renderer = 'canvas';
      node6.removeChild( path14 );
      node3.renderer = null;
      path14.renderer = null;
      scene.insertChild( 0, node5 );
      path10.insertChild( 2, scene );
      path10.insertChild( 2, path14 );
      path10.removeChild( path13 );
      path10.removeChild( scene );
      node6.insertChild( 0, path15 );
      path12.insertChild( 2, path11 );
      path15.renderer = 'canvas';
      path11.insertChild( 0, node6 );
      node5.renderer = null;
      path16.insertChild( 0, path17 );
      scene.removeChild( node5 );
      node6.removeChild( path15 );
      path10.renderer = 'canvas';
      path12.insertChild( 3, path16 );
      node7.renderer = null;
      path15.insertChild( 0, path17 );
      path14.renderer = 'canvas';
      path14.insertChild( 0, node6 );
      path10.removeChild( node7 );
      path12.insertChild( 3, node5 );
      node7.insertChild( 0, path13 );
      node7.insertChild( 0, node3 );
      path10.removeChild( node4 );
      path15.removeChild( path17 );
      node5.insertChild( 0, path17 );
      path11.insertChild( 0, scene );
      node7.insertChild( 2, path17 );
      path8.renderer = null;
      path14.removeChild( node6 );
      path13.renderer = 'canvas';
      node7.insertChild( 1, node6 );
      path16.renderer = null;
      path9.renderer = 'svg';
      path13.insertChild( 0, path16 );
      node5.removeChild( path17 );
      node7.insertChild( 0, path14 );
      node5.insertChild( 0, scene );
      node5.renderer = 'svg';
      path10.insertChild( 0, node5 );
      path17.renderer = 'svg';
      path12.removeChild( path10 );
      node3.removeChild( node5 );
      path16.insertChild( 0, node4 );
      path15.insertChild( 0, path17 );
      path8.removeChild( path16 );
      path17.renderer = 'canvas';
      node7.insertChild( 3, path8 );
      path15.removeChild( path17 );
      path9.renderer = 'svg';
      node3.renderer = null;
      node5.insertChild( 0, path9 );
      path8.insertChild( 0, path10 );
      node3.insertChild( 0, node4 );
      path17.removeChild( scene );
      path10.renderer = 'svg';
      path11.renderer = null;
      scene.insertChild( 0, path9 );
      path16.removeChild( path17 );
      node7.renderer = null;
      path12.removeChild( node3 );
      path8.renderer = null;
      path13.insertChild( 3, node4 );
      path12.removeChild( node5 );
      node7.removeChild( node3 );
      node3.removeChild( node4 );
      scene.removeChild( path9 );
      path12.renderer = 'canvas';
      node7.removeChild( node5 );
      path8.removeChild( path10 );
      node5.insertChild( 1, node4 );
      path9.renderer = 'canvas';
      path9.insertChild( 0, path14 );
      node5.insertChild( 3, node3 );
      path11.insertChild( 1, node4 );
      path8.insertChild( 0, path17 );
      path12.removeChild( path11 );
      path14.renderer = null;
      path16.removeChild( node4 );
      node4.renderer = null;
      node3.renderer = 'svg';
      path8.insertChild( 0, path13 );
      path17.insertChild( 0, node4 );
      scene.insertChild( 0, path15 );
      path15.insertChild( 0, node6 );
      node6.renderer = 'svg';
      node3.renderer = 'svg';
      node5.insertChild( 1, path17 );
      path14.renderer = null;
      node3.renderer = null;
      path17.removeChild( node4 );
      path17.insertChild( 0, path11 );
      scene.insertChild( 0, path14 );
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var rotatedSquaresHardcodedXY = {
    name: 'Rotated group of squares with hardcoded xy',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      for( var i = 0; i < 4000; i++ ) {
        scene.addChild( new scenery.Rectangle( ( Math.PI * i ) % this.width, ( 27 * i ) % this.height, 20, 20, {
          fill: 'rgba(255,0,0,0.3)',
          stroke: '#000000'
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      this.scene.rotateAround( new dot.Vector2( this.width / 2, this.height / 2 ), 0.1 );
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var rotatedSquaresTransformXY = {
    name: 'Rotated group of squares with transform xy',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      for( var i = 0; i < 4000; i++ ) {
        scene.addChild( new scenery.Rectangle( 0, 0, 20, 20, {
          fill: 'rgba(0,0,255,0.3)',
          stroke: '#000000',
          x: ( Math.PI * i ) % this.width,
          y: ( 27 * i ) % this.height
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      this.scene.rotateAround( new dot.Vector2( this.width / 2, this.height / 2 ), 0.1 );
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var rotatedSquaresIndividual = {
    name: 'Individually Rotated Squares',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      for( var i = 0; i < 4000; i++ ) {
        scene.addChild( new scenery.Rectangle( 0, 0, 20, 20, {
          fill: 'rgba(0,255,0,0.3)',
          stroke: '#000000',
          x: ( Math.PI * i ) % this.width,
          y: ( 27 * i ) % this.height
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      _.each( this.scene.getChildren(), function( child ) {
        child.rotate( 0.1 );
      } );
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var fastSquaresCanvas = {
    name: 'Fast Squares Canvas',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      for( var i = 0; i < 1000; i++ ) {
        scene.addChild( new scenery.Rectangle( 0, 0, 20, 20, {
          fill: 'rgba(0,255,0,0.3)',
          stroke: '#000000',
          x: ( Math.PI * i ) % this.width,
          y: ( 27 * i ) % this.height
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
      
      this.iterationCount = 0;
    },
    step: function() {
      this.iterationCount++;
      var children = this.scene.getChildren();
      for ( var i = 0; i < children.length; i++ ) {
        var child = children[i];
        if ( i % 3 ) {
          child.rotate( 0.1 );
        } else {
          child.fill = ( this.iterationCount + i ) % 2 === 0 ? 'rgba(0,255,0,0.3)' : 'rgba(0,0,255,0.3)';
        }
      }
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var fastSquaresSVG = {
    name: 'Fast Squares SVG',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main, { renderer: 'svg' } );
      for( var i = 0; i < 500; i++ ) {
        scene.addChild( new scenery.Rectangle( 0, 0, 20, 20, {
          fill: 'rgba(0,255,0,0.3)',
          stroke: '#000000',
          x: ( Math.PI * i ) % this.width,
          y: ( 27 * i ) % this.height
        } ) );
      }
      
      this.main = main;
      this.scene = scene;
      
      this.iterationCount = 0;
    },
    step: function() {
      this.iterationCount++;
      var children = this.scene.getChildren();
      for ( var i = 0; i < children.length; i++ ) {
        var child = children[i];
        if ( i % 3 ) {
          child.rotate( 0.1 );
        } else {
          child.fill = ( this.iterationCount + i ) % 2 === 0 ? 'rgba(0,255,0,0.3)' : 'rgba(0,0,255,0.3)';
        }
      }
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  var addingHexagons = {
    name: 'Adding Hexagons',
    count: 20,
    before: function() {
      this.width = 1024;
      this.height = 768;
      
      this.x = 0;
      this.y = 0;
      
      var main = $( '#main' );
      main.width( this.width );
      main.height( this.height );
      var scene = new scenery.Scene( main );
      
      this.main = main;
      this.scene = scene;
    },
    step: function() {
      for ( var i = 0; i < 200; i++ ) {
        this.scene.addChild( new scenery.Path( {
          shape: kite.Shape.regularPolygon( 6, 22 ),
          fill: 'rgba(255,0,255,0.3)',
          stroke: '#000000',
          x: this.x,
          y: this.y
        } ) );
        this.x = ( this.x + 213 ) % this.width;
        this.y = ( this.y + 377 ) % this.height;
      }
      this.scene.updateScene();
    },
    after: function() {
      this.main.empty();
    }
  };
  
  marks.currentMarks = [
    // textNodeInstances,
    // textPathInstances,
    textBoundsChanges,
    fuzzRecordAddRemoveRender,
    rotatedSquaresHardcodedXY,
    rotatedSquaresTransformXY,
    rotatedSquaresIndividual,
    addingHexagons,
    fastSquaresCanvas,
    fastSquaresSVG
    // {
    //   name: 'Empty Loop'
    // }
  ];
  
})();
