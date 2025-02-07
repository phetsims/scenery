// Copyright 2024-2025, University of Colorado Boulder

/**
 * A HighlightPath with the default styling for group highlights. Group highlights typically surround
 * a group of components that have one stop in the traversal order and are navigated by arrow keys. For example,
 * a radio button group or menu list.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Shape from '../../../kite/js/Shape.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import type { HighlightPathOptions } from '../accessibility/HighlightPath.js';
import HighlightPath from '../accessibility/HighlightPath.js';
import scenery from '../scenery.js';

type SelfOptions = EmptySelfOptions;
export type GroupHighlightPathOptions = HighlightPathOptions & SelfOptions;

export default class GroupHighlightPath extends HighlightPath {
  public constructor( shape: Shape | string | null, providedOptions?: GroupHighlightPathOptions ) {
    const options = optionize<GroupHighlightPathOptions, SelfOptions, HighlightPathOptions>()( {
      outerStroke: HighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR,
      innerStroke: HighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR,
      outerLineWidth: HighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: HighlightPath.GROUP_INNER_LINE_WIDTH
    }, providedOptions );

    super( shape, options );
  }
}

scenery.register( 'GroupHighlightPath', GroupHighlightPath );