// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for SCENERY/Image (not the Image HTMLElement)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import VoidIO from '../../../tandem/js/types/VoidIO.js';
import scenery from '../scenery.js';
import Image from './Image.js';
import NodeIO from './NodeIO.js';

class ImageIO extends NodeIO {}

ImageIO.methods = {

  setImage: {
    returnType: VoidIO,
    parameterTypes: [ StringIO ],
    implementation: function( base64Text ) {
      const im = new window.Image();
      im.src = base64Text;
      this.image = im;
    },
    documentation: 'Set the image from a base64 string',
    invocableForReadOnlyElements: false
  }
};

ImageIO.documentation = 'The tandem IO Type for the scenery Text node';
ImageIO.events = [ 'changed' ];
ImageIO.validator = { valueType: Image };
ImageIO.typeName = 'ImageIO';
ObjectIO.validateIOType( ImageIO );

scenery.register( 'ImageIO', ImageIO );
export default ImageIO;