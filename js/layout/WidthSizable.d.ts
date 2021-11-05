// Copyright 2021, University of Colorado Boulder

import TinyProperty from '../../../axon/js/TinyProperty.js';
declare type Constructor<T = {}> = new ( ...args: any[] ) => T;
declare const WidthSizable: <SuperType extends Constructor>( key: SuperType, ...args: any[] ) => {
    new ( ...args: any[] ): {
        preferredWidthProperty: TinyProperty;
        minimumWidthProperty: TinyProperty;
        widthSizable: true;
        preferredWidth: number | null;
        minimumWidth: number | null;
    };
} & SuperType;
export default WidthSizable;