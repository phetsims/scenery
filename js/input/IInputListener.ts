// Copyright 2021, University of Colorado Boulder

import { SceneryEvent } from '../imports.js';

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

interface IInputListener {
  interrupt?: () => void;
  cursor?: string | null;

  focus?: ( event: SceneryEvent ) => void;
  blur?: ( event: SceneryEvent ) => void;
  focusin?: ( event: SceneryEvent ) => void;
  focusout?: ( event: SceneryEvent ) => void;

  keydown?: ( event: SceneryEvent ) => void;
  keyup?: ( event: SceneryEvent ) => void;

  click?: ( event: SceneryEvent ) => void;
  input?: ( event: SceneryEvent ) => void;
  change?: ( event: SceneryEvent ) => void;

  down?: ( event: SceneryEvent ) => void;
  mousedown?: ( event: SceneryEvent ) => void;
  touchdown?: ( event: SceneryEvent ) => void;
  pendown?: ( event: SceneryEvent ) => void;

  up?: ( event: SceneryEvent ) => void;
  mouseup?: ( event: SceneryEvent ) => void;
  touchup?: ( event: SceneryEvent ) => void;
  penup?: ( event: SceneryEvent ) => void;

  cancel?: ( event: SceneryEvent ) => void;
  mousecancel?: ( event: SceneryEvent ) => void;
  touchcancel?: ( event: SceneryEvent ) => void;
  pencancel?: ( event: SceneryEvent ) => void;

  move?: ( event: SceneryEvent ) => void;
  mousemove?: ( event: SceneryEvent ) => void;
  touchmove?: ( event: SceneryEvent ) => void;
  penmove?: ( event: SceneryEvent ) => void;

  wheel?: ( event: SceneryEvent ) => void;
  mousewheel?: ( event: SceneryEvent ) => void;
  touchwheel?: ( event: SceneryEvent ) => void;
  penwheel?: ( event: SceneryEvent ) => void;

  enter?: ( event: SceneryEvent ) => void;
  mouseenter?: ( event: SceneryEvent ) => void;
  touchenter?: ( event: SceneryEvent ) => void;
  penenter?: ( event: SceneryEvent ) => void;

  exit?: ( event: SceneryEvent ) => void;
  mouseexit?: ( event: SceneryEvent ) => void;
  touchexit?: ( event: SceneryEvent ) => void;
  penexit?: ( event: SceneryEvent ) => void;

  over?: ( event: SceneryEvent ) => void;
  mouseover?: ( event: SceneryEvent ) => void;
  touchover?: ( event: SceneryEvent ) => void;
  penover?: ( event: SceneryEvent ) => void;

  out?: ( event: SceneryEvent ) => void;
  mouseout?: ( event: SceneryEvent ) => void;
  touchout?: ( event: SceneryEvent ) => void;
  penout?: ( event: SceneryEvent ) => void;
}

export default IInputListener;