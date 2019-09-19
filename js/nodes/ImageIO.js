// Copyright 2017-2019, University of Colorado Boulder

/**
 * IO type for SCENERY/Image (not the Image HTMLElement)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const Image = require( 'SCENERY/nodes/Image' );
  const NodeIO = require( 'SCENERY/nodes/NodeIO' );
  const ObjectIO = require( 'TANDEM/types/ObjectIO' );
  const scenery = require( 'SCENERY/scenery' );
  const StringIO = require( 'TANDEM/types/StringIO' );
  const VoidIO = require( 'TANDEM/types/VoidIO' );

  class ImageIO extends NodeIO {}

  ImageIO.methods = {

    setImage: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( base64Text ) {
        const im = new window.Image();
        im.src = base64Text;
        this.phetioObject.image = im;
      },
      documentation: 'Set the image from a base64 string',
      invocableForReadOnlyElements: false
    }
  };

  ImageIO.documentation = 'The tandem IO type for the scenery Text node';
  ImageIO.events = [ 'changed' ];
  ImageIO.validator = { valueType: Image };
  ImageIO.typeName = 'ImageIO';
  ObjectIO.validateSubtype( ImageIO );

  return scenery.register( 'ImageIO', ImageIO );
} );
