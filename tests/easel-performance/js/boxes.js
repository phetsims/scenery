
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    // constants
    var boxSizeRatio = 0.75;
    var boxTotalSize = 200;
    
    function buildEaselStage( main ) {
        var canvas = document.createElement( 'canvas' );
        canvas.id = 'easel-canvas';
        canvas.width = main.width();
        canvas.height = main.height();
        main.append( canvas );

        return new createjs.Stage( canvas );
    }
    
    function buildBaseContext( main ) {
        var baseCanvas = document.createElement( 'canvas' );
        baseCanvas.id = 'base-canvas';
        baseCanvas.width = main.width();
        baseCanvas.height = main.height();
        main.append( baseCanvas );
        
        return phet.canvas.initCanvas( baseCanvas );
    }    
    
    phet.tests.easelVariableBox = function( main, resolution ) {
        var stage = buildEaselStage( main );
        var grid = new createjs.Container();
        
        var size = boxTotalSize;
        
        var boxRadius = 0.5 * boxSizeRatio * size / resolution;
        
        for( var row = 0; row < resolution; row++ ) {
            for( var col = 0; col < resolution; col++ ) {
                var shape = new createjs.Shape();
                shape.graphics.beginFill('rgba(255,0,0,1)').drawRect( -boxRadius, -boxRadius, boxRadius * 2, boxRadius * 2 );
                
                shape.x = ( col - ( resolution - 1 ) / 2 ) * size / resolution;
                shape.y = ( row - ( resolution - 1 ) / 2 ) * size / resolution;
                
                grid.addChild( shape );
            }
        }

        // center the grid        
        grid.x = main.width() / 2;
        grid.y = main.height() / 2;
        stage.addChild( grid );
        
        // return step function
        return function( timeElapsed ) {
            grid.rotation += timeElapsed * 180 / Math.PI;
            stage.update();
        }
    }

    phet.tests.customVariableBox = function( main, resolution ) {
        var baseContext = buildBaseContext( main );
        
        var xCenter = main.width() / 2;
        var yCenter = main.height() / 2;
        var rotation = 0;
        baseContext.fillStyle = 'rgba(255,0,0,1)';
        
        var size = boxTotalSize;
        
        var boxRadius = 0.5 * boxSizeRatio / resolution;
        var spotWidth = 1 / resolution;
        
        // TODO: optimize this
        var halfIshResolution = ( resolution - 1 ) / 2;
        
        // return step function
        return function( timeElapsed ) {
            // clear old location
            baseContext.clearRect( xCenter - 150, yCenter - 150, 300, 300 );
            // TODO: consider whether clearRect is faster if it's not under a transform!
            
            baseContext.save();
            rotation += timeElapsed;
            baseContext.translate( xCenter, yCenter );
            baseContext.rotate( rotation );
            baseContext.scale( size, size );
            
            baseContext.beginPath();
            
            for( var row = 0; row < resolution; row++ ) {
                var baseY = ( row - halfIshResolution ) / resolution;
                for( var col = 0; col < resolution; col++ ) {
                    var baseX = ( col - halfIshResolution ) / resolution;
                    baseContext.fillRect( baseX - boxRadius, baseY - boxRadius, boxRadius * 2, boxRadius * 2 );
                    // baseContext.moveTo( baseX - boxRadius, baseY - boxRadius );
                    // baseContext.lineTo( baseX + boxRadius, baseY - boxRadius );
                    // baseContext.lineTo( baseX + boxRadius, baseY + boxRadius );
                    // baseContext.lineTo( baseX - boxRadius, baseY + boxRadius );
                    // baseContext.lineTo( baseX - boxRadius, baseY - boxRadius );
                }
            }
            // baseContext.fill();
            
            baseContext.restore();
        }
    }
    
    phet.tests.sceneVariableBox = function( main, resolution ) {
        var baseContext = buildBaseContext( main );
        
        var size = boxTotalSize;
        
        var boxRadius = 0.5 * boxSizeRatio * size / resolution;
        
        var grid = new phet.scene.Node();
        
        for( var row = 0; row < resolution; row++ ) {
            for( var col = 0; col < resolution; col++ ) {
                grid.addChild( new phet.scene.nodes.Rectangle({
                    x: ( col - ( resolution - 1 ) / 2 ) * size / resolution - boxRadius,
                    y: ( row - ( resolution - 1 ) / 2 ) * size / resolution - boxRadius,
                    width: boxRadius * 2,
                    height: boxRadius * 2,
                    fill: 'rgba(255,0,0,1)'
                }) );
            }
        }
        
        // center the grid
        grid.translate( main.width() / 2, main.height() / 2 );

        // return step function
        return function( timeElapsed ) {
            grid.rotate( timeElapsed );
            
            baseContext.clearRect( main.width() / 2 - 150, main.height() / 2 - 150, 300, 300 );
            // baseContext.clearRect( 0, 0, main.width(), main.height() );
            var state = new phet.scene.RenderState();
            state.context = baseContext;
            grid.render( state );
        }
    }
})();
